package main

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"strings"
)

const namesURL = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"

const (
	NLayer    = 1
	NEmbd    = 16
	BlockSize = 16
	NHead     = 4
	HeadDim   = NEmbd / NHead
	LearnRate = 0.01
	Beta1     = 0.85
	Beta2     = 0.99
	EpsAdam   = 1e-8
	NumSteps  = 1000
)

var rng = rand.New(rand.NewSource(42))

func RandN(mu, sigma float64) float64 {
	return mu + sigma*rng.NormFloat64()
}

type Value struct {
	Data      float64
	Grad      float64
	Children  []*Value
	LocalGrads []float64
}

func NewValue(data float64, children []*Value, localGrads []float64) *Value {
	return &Value{
		Data:      data,
		Grad:      0,
		Children:  children,
		LocalGrads: localGrads,
	}
}

func Add(a, b *Value) *Value {
	return NewValue(a.Data + b.Data, []*Value{a, b}, []float64{1, 1})
}

func Mul(a, b *Value) *Value {
	return NewValue(a.Data * b.Data, []*Value{a, b}, []float64{b.Data, a.Data})
}

func Neg(a *Value) *Value {
	n1 := NewValue(-1, nil, nil)
	return Mul(a, n1)
}

func Sub(a, b *Value) *Value {
	return Add(a, Neg(b))
}

func Div(a, b *Value) *Value {
	inv := NewValue(-1, nil, nil)
	return Mul(a, Pow(b, inv))
}

func Pow(a, b *Value) *Value {
	if b.Data == 0 {
		return NewValue(1, nil, nil)
	}
	p := math.Pow(a.Data, b.Data)
	dpdself := b.Data * math.Pow(a.Data, b.Data-1)
	return NewValue(p, []*Value{a}, []float64{dpdself})
}

func Log(a *Value) *Value {
	l := math.Log(a.Data)
	return NewValue(l, []*Value{a}, []float64{1 / a.Data})
}

func Exp(a *Value) *Value {
	e := math.Exp(a.Data)
	return NewValue(e, []*Value{a}, []float64{e})
}

func ReLU(a *Value) *Value {
	d := math.Max(0, a.Data)
	g := float64(0)
	if a.Data > 0 {
		g = 1
	}
	return NewValue(d, []*Value{a}, []float64{g})
}

func (v *Value) Backward() {
	topo := []*Value{}
	visited := make(map[*Value]struct{})
	var buildTopo func(*Value)
	buildTopo = func(node *Value) {
		if _, ok := visited[node]; ok {
			return
		}
		visited[node] = struct{}{}
		for _, child := range node.Children {
			buildTopo(child)
		}
		topo = append(topo, node)
	}
	buildTopo(v)
	v.Grad = 1.0
	for i := len(topo) - 1; i >= 0; i-- {
		node := topo[i]
		for j, child := range node.Children {
			child.Grad += node.LocalGrads[j] * node.Grad
		}
	}
}

type Vec []*Value

type Matrix [][]*Value

func VecAdd(a, b Vec) Vec {
	res := make(Vec, len(a))
	for i := range a {
		res[i] = Add(a[i], b[i])
	}
	return res
}

func VecCopy(v Vec) Vec {
	res := make(Vec, len(v))
	copy(res, v)
	return res
}

func VecReLU(v Vec) Vec {
	res := make(Vec, len(v))
	for i, vi := range v {
		res[i] = ReLU(vi)
	}
	return res
}

func ReduceAdd(vs Vec) *Value {
	if len(vs) == 0 {
		return NewValue(0, nil, nil)
	}
	sum := vs[0]
	for i := 1; i < len(vs); i++ {
		sum = Add(sum, vs[i])
	}
	return sum
}

func Linear(w Matrix, x Vec) Vec {
	res := make(Vec, len(w))
	for i := range w {
		sum := NewValue(0, nil, nil)
		for j := range w[i] {
			sum = Add(sum, Mul(w[i][j], x[j]))
		}
		res[i] = sum
	}
	return res
}

func RMSNorm(x Vec) Vec {
	n := len(x)
	invN := NewValue(1.0 / float64(n), nil, nil)
	msTerms := make(Vec, n)
	for i, xi := range x {
		msTerms[i] = Mul(xi, xi)
	}
	ms := Mul(ReduceAdd(msTerms), invN)
	epsV := NewValue(1e-5, nil, nil)
	scaleBase := Add(ms, epsV)
	expV := NewValue(-0.5, nil, nil)
	scale := Pow(scaleBase, expV)
	res := make(Vec, n)
	for i, xi := range x {
		res[i] = Mul(xi, scale)
	}
	return res
}

func Softmax(logits Vec) Vec {
	n := len(logits)
	if n == 0 {
		return Vec{}
	}
	maxData := logits[0].Data
	for _, l := range logits {
		if l.Data > maxData {
			maxData = l.Data
		}
	}
	maxVal := NewValue(maxData, nil, nil)
	exps := make(Vec, n)
	for i, logit := range logits {
		exps[i] = Exp(Sub(logit, maxVal))
	}
	total := ReduceAdd(exps)
	res := make(Vec, n)
	for i, e := range exps {
		res[i] = Div(e, total)
	}
	return res
}

func MakeMatrix(nout, nin int, std float64) Matrix {
	res := make(Matrix, nout)
	for i := 0; i < nout; i++ {
		res[i] = make([]*Value, nin)
		for j := 0; j < nin; j++ {
			res[i][j] = NewValue(RandN(0, std), nil, nil)
		}
	}
	return res
}

func GPT(tokenID, posID int, keys [][]Vec, values [][]Vec, state map[string]Matrix) Vec {
	tokEmb := state["wte"][tokenID]
	posEmb := state["wpe"][posID]
	x := VecAdd(tokEmb, posEmb)
	x = RMSNorm(x)
	for li := 0; li < NLayer; li++ {
		// attention block
		xRes := VecCopy(x)
		x = RMSNorm(x)
		wqKey := fmt.Sprintf("layer%d.attn_wq", li)
		wkKey := fmt.Sprintf("layer%d.attn_wk", li)
		wvKey := fmt.Sprintf("layer%d.attn_wv", li)
		woKey := fmt.Sprintf("layer%d.attn_wo", li)
		q := Linear(state[wqKey], x)
		k := Linear(state[wkKey], x)
		v := Linear(state[wvKey], x)
		keys[li] = append(keys[li], k)
		values[li] = append(values[li], v)
		xAttn := Vec{}
		for h := 0; h < NHead; h++ {
			hs := h * HeadDim
			qh := q[hs : hs+HeadDim]
			numPast := len(keys[li])
			kh := make([]Vec, numPast)
			for t := 0; t < numPast; t++ {
				kh[t] = keys[li][t][hs : hs+HeadDim]
			}
			vh := make([]Vec, numPast)
			for t := 0; t < numPast; t++ {
				vh[t] = values[li][t][hs : hs+HeadDim]
			}
			attnLogits := make(Vec, numPast)
			scale := 1.0 / math.Sqrt(float64(HeadDim))
			scaleV := NewValue(scale, nil, nil)
			for t := 0; t < numPast; t++ {
				dot := NewValue(0, nil, nil)
				for j := 0; j < HeadDim; j++ {
					dot = Add(dot, Mul(qh[j], kh[t][j]))
				}
				attnLogits[t] = Mul(dot, scaleV)
			}
			attnW := Softmax(attnLogits)
			headOut := make(Vec, HeadDim)
			for j := 0; j < HeadDim; j++ {
				sumOut := NewValue(0, nil, nil)
				for t := 0; t < numPast; t++ {
					sumOut = Add(sumOut, Mul(attnW[t], vh[t][j]))
				}
				headOut[j] = sumOut
			}
			xAttn = append(xAttn, headOut...)
		}
		x = Linear(state[woKey], xAttn)
		x = VecAdd(x, xRes)
		// MLP block
		xRes = VecCopy(x)
		x = RMSNorm(x)
		fc1Key := fmt.Sprintf("layer%d.mlp_fc1", li)
		fc2Key := fmt.Sprintf("layer%d.mlp_fc2", li)
		x = Linear(state[fc1Key], x)
		x = VecReLU(x)
		x = Linear(state[fc2Key], x)
		x = VecAdd(x, xRes)
	}
	logits := Linear(state["lm_head"], x)
	return logits
}

func Categorical(probs Vec) int {
	ws := make([]float64, len(probs))
	for i, p := range probs {
		ws[i] = p.Data
	}
	r := rng.Float64()
	csum := 0.0
	for i, w := range ws {
		csum += w
		if r < csum {
			return i
		}
	}
	return len(ws) - 1 // fallback
}

func ensureInput() error {
	if _, err := os.Stat("input.txt"); err == nil {
		return nil
	}
	resp, err := http.Get(namesURL)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	f, err := os.Create("input.txt")
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(f, resp.Body)
	return err
}

func main() {
	if err := ensureInput(); err != nil {
		fmt.Printf("Error downloading data: %v\n", err)
		return
	}
	file, err := os.Open("input.txt")
	if err != nil {
		fmt.Printf("Error opening input.txt: %v\n", err)
		return
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	docs := []string{}
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			docs = append(docs, line)
		}
	}
	rand.Shuffle(len(docs), func(i, j int) {
		docs[i], docs[j] = docs[j], docs[i]
	})
	fmt.Printf("num docs: %d\n", len(docs))
	// build vocab
	charSet := make(map[rune]struct{})
	for _, doc := range docs {
		for _, ch := range []rune(doc) {
			charSet[ch] = struct{}{}
		}
	}
	chars := make([]rune, 0, len(charSet))
	for ch := range charSet {
		chars = append(chars, ch)
	}
	sort.Slice(chars, func(i, j int) bool {
		return chars[i] < chars[j]
	})
	vocabSize := len(chars) + 1
	BOS := vocabSize - 1
	charToID := make(map[rune]int)
	for i, ch := range chars {
		charToID[ch] = i
	}
	fmt.Printf("vocab size: %d\n", vocabSize)
	// init params
	state := make(map[string]Matrix)
	const std = 0.08
	state["wte"] = MakeMatrix(vocabSize, NEmbd, std)
	state["wpe"] = MakeMatrix(BlockSize, NEmbd, std)
	state["lm_head"] = MakeMatrix(vocabSize, NEmbd, std)
	for i := 0; i < NLayer; i++ {
		state[fmt.Sprintf("layer%d.attn_wq", i)] = MakeMatrix(NEmbd, NEmbd, std)
		state[fmt.Sprintf("layer%d.attn_wk", i)] = MakeMatrix(NEmbd, NEmbd, std)
		state[fmt.Sprintf("layer%d.attn_wv", i)] = MakeMatrix(NEmbd, NEmbd, std)
		state[fmt.Sprintf("layer%d.attn_wo", i)] = MakeMatrix(NEmbd, NEmbd, std)
		state[fmt.Sprintf("layer%d.mlp_fc1", i)] = MakeMatrix(4*NEmbd, NEmbd, std)
		state[fmt.Sprintf("layer%d.mlp_fc2", i)] = MakeMatrix(NEmbd, 4*NEmbd, std)
	}
	params := []*Value{}
	for _, mat := range state {
		for _, row := range mat {
			params = append(params, row...)
		}
	}
	fmt.Printf("num params: %d\n", len(params))
	// Adam buffers
	m := make([]float64, len(params))
	vAdam := make([]float64, len(params))
	// training
	for step := 0; step < NumSteps; step++ {
		docIdx := step % len(docs)
		doc := docs[docIdx]
		tokens := []int{BOS}
		for _, ch := range []rune(doc) {
			tokens = append(tokens, charToID[ch])
		}
		tokens = append(tokens, BOS)
		n := BlockSize
		if len(tokens)-1 < n {
			n = len(tokens) - 1
		}
		keys := make([][]Vec, NLayer)
		values := make([][]Vec, NLayer)
		for i := range keys {
			keys[i] = []Vec{}
			values[i] = []Vec{}
		}
		losses := Vec{}
		for pos := 0; pos < n; pos++ {
			tokenID := tokens[pos]
			targetID := tokens[pos+1]
			logits := GPT(tokenID, pos, keys, values, state)
			probs := Softmax(logits)
			logProb := Log(probs[targetID])
			lossT := Neg(logProb)
			losses = append(losses, lossT)
		}
		invN := NewValue(1.0/float64(n), nil, nil)
		loss := Mul(ReduceAdd(losses), invN)
		loss.Backward()
		lrT := LearnRate * (1 - float64(step)/float64(NumSteps))
		for i := range params {
			p := params[i]
			grad := p.Grad
			m[i] = Beta1*m[i] + (1-Beta1)*grad
			vAdam[i] = Beta2*vAdam[i] + (1-Beta2)*grad*grad
			mHat := m[i] / (1 - math.Pow(Beta1, float64(step+1)))
			vHat := vAdam[i] / (1 - math.Pow(Beta2, float64(step+1)))
			p.Data -= lrT * mHat / (math.Sqrt(vHat) + EpsAdam)
			p.Grad = 0
		}
		fmt.Printf("step %4d / %4d | loss %.4f\r", step+1, NumSteps, loss.Data)
	}
	fmt.Println()
	// inference
	fmt.Println("--- inference (new, hallucinated names) ---")
	temp := 0.5
	for sidx := 0; sidx < 20; sidx++ {
		keys := make([][]Vec, NLayer)
		values := make([][]Vec, NLayer)
		for i := range keys {
			keys[i] = []Vec{}
			values[i] = []Vec{}
		}
		sample := []rune{}
		tokenID := BOS
		for pos := 0; pos < BlockSize; pos++ {
			logits := GPT(tokenID, pos, keys, values, state)
			invTemp := 1.0 / temp
			tempLogits := Vec{}
			invTempV := NewValue(invTemp, nil, nil)
			for _, l := range logits {
				tempLogits = append(tempLogits, Mul(l, invTempV))
			}
			probs := Softmax(tempLogits)
			nextToken := Categorical(probs)
			tokenID = nextToken
			if nextToken == BOS {
				break
			}
			if nextToken < len(chars) {
				sample = append(sample, chars[nextToken])
			}
		}
		fmt.Printf("sample %2d: %s\n", sidx+1, string(sample))
	}
}