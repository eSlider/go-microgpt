package main

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"runtime"
	"runtime/pprof"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
)

const namesURL = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"

const (
	NLayer    = 1
	NEmbd     = 16
	BlockSize = 16
	NHead     = 4
	HeadDim   = NEmbd / NHead
	LearnRate = 0.01
	Beta1     = 0.85
	Beta2     = 0.99
	EpsAdam   = 1e-8
	NumSteps  = 1000

	// Parallelism only helps when per-task work is large enough.
	MinParallelQKVWork  = 1024
	MinParallelHeadWork = 128
)

var rng = rand.New(rand.NewSource(42))
var backwardMarkCounter uint64
var valuePool = sync.Pool{
	New: func() any {
		return &Value{}
	},
}

func RandN(mu, sigma float64) float64 {
	return mu + sigma*rng.NormFloat64()
}

type Value struct {
	Data       float64
	Grad       float64
	Children   []*Value
	LocalGrads []float64
	isParam    bool
	mark       uint64
}

func NewValue(data float64, children []*Value, localGrads []float64) *Value {
	v := valuePool.Get().(*Value)
	v.Data = data
	v.Grad = 0
	v.Children = children
	v.LocalGrads = localGrads
	v.isParam = false
	v.mark = 0
	return v
}

func NewParamValue(data float64) *Value {
	return &Value{
		Data:    data,
		Grad:    0,
		isParam: true,
	}
}

func Add(a, b *Value) *Value {
	return NewValue(a.Data+b.Data, []*Value{a, b}, []float64{1, 1})
}

func Mul(a, b *Value) *Value {
	return NewValue(a.Data*b.Data, []*Value{a, b}, []float64{b.Data, a.Data})
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

func (v *Value) Backward() []*Value {
	topo := []*Value{}
	markID := atomic.AddUint64(&backwardMarkCounter, 1)
	var buildTopo func(*Value)
	buildTopo = func(node *Value) {
		if node.mark == markID {
			return
		}
		node.mark = markID
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
	return topo
}

func releaseGraph(topo []*Value) {
	for _, node := range topo {
		if node.isParam {
			continue
		}
		node.Data = 0
		node.Grad = 0
		node.Children = nil
		node.LocalGrads = nil
		node.mark = 0
		valuePool.Put(node)
	}
}

type Vec []*Value

type Matrix [][]*Value

type headResult struct {
	idx int
	out Vec
}

type layerKeySet struct {
	wq  string
	wk  string
	wv  string
	wo  string
	fc1 string
	fc2 string
}

type inferenceLayerState struct {
	wq  [][]float64
	wk  [][]float64
	wv  [][]float64
	wo  [][]float64
	fc1 [][]float64
	fc2 [][]float64
}

type inferenceState struct {
	wte    [][]float64
	wpe    [][]float64
	lmHead [][]float64
	layers []inferenceLayerState
}

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

func Dot(a, b Vec) *Value {
	n := len(a)
	children := make([]*Value, 2*n)
	localGrads := make([]float64, 2*n)
	sum := 0.0
	for i := range n {
		av := a[i]
		bv := b[i]
		sum += av.Data * bv.Data
		children[i] = av
		localGrads[i] = bv.Data
		children[n+i] = bv
		localGrads[n+i] = av.Data
	}
	return NewValue(sum, children, localGrads)
}

func WeightedSum(weights Vec, vectors []Vec, dim int) *Value {
	n := len(weights)
	children := make([]*Value, 2*n)
	localGrads := make([]float64, 2*n)
	sum := 0.0
	for i := range n {
		w := weights[i]
		v := vectors[i][dim]
		sum += w.Data * v.Data
		children[i] = w
		localGrads[i] = v.Data
		children[n+i] = v
		localGrads[n+i] = w.Data
	}
	return NewValue(sum, children, localGrads)
}

func Linear(w Matrix, x Vec) Vec {
	res := make(Vec, len(w))
	for i := range w {
		res[i] = Dot(w[i], x)
	}
	return res
}

func RMSNorm(x Vec) Vec {
	n := len(x)
	invN := NewValue(1.0/float64(n), nil, nil)
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
	for i := range nout {
		res[i] = make([]*Value, nin)
		for j := range nin {
			res[i][j] = NewParamValue(RandN(0, std))
		}
	}
	return res
}

func toFloatMatrix(mat Matrix) [][]float64 {
	out := make([][]float64, len(mat))
	for i := range mat {
		row := make([]float64, len(mat[i]))
		for j := range mat[i] {
			row[j] = mat[i][j].Data
		}
		out[i] = row
	}
	return out
}

func buildInferenceState(state map[string]Matrix, layerKeys []layerKeySet) inferenceState {
	inf := inferenceState{
		wte:    toFloatMatrix(state["wte"]),
		wpe:    toFloatMatrix(state["wpe"]),
		lmHead: toFloatMatrix(state["lm_head"]),
		layers: make([]inferenceLayerState, len(layerKeys)),
	}
	for i := range layerKeys {
		ks := layerKeys[i]
		inf.layers[i] = inferenceLayerState{
			wq:  toFloatMatrix(state[ks.wq]),
			wk:  toFloatMatrix(state[ks.wk]),
			wv:  toFloatMatrix(state[ks.wv]),
			wo:  toFloatMatrix(state[ks.wo]),
			fc1: toFloatMatrix(state[ks.fc1]),
			fc2: toFloatMatrix(state[ks.fc2]),
		}
	}
	return inf
}

func vecAddF(a, b []float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

func linearF(w [][]float64, x []float64) []float64 {
	out := make([]float64, len(w))
	for i := range w {
		sum := 0.0
		row := w[i]
		for j := range row {
			sum += row[j] * x[j]
		}
		out[i] = sum
	}
	return out
}

func rmsNormF(x []float64) []float64 {
	ms := 0.0
	for _, v := range x {
		ms += v * v
	}
	ms /= float64(len(x))
	scale := math.Pow(ms+1e-5, -0.5)
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = v * scale
	}
	return out
}

func reluF(x []float64) []float64 {
	out := make([]float64, len(x))
	for i, v := range x {
		if v > 0 {
			out[i] = v
		}
	}
	return out
}

func softmaxF(logits []float64) []float64 {
	if len(logits) == 0 {
		return []float64{}
	}
	maxV := logits[0]
	for _, v := range logits[1:] {
		if v > maxV {
			maxV = v
		}
	}
	exps := make([]float64, len(logits))
	sum := 0.0
	for i, v := range logits {
		e := math.Exp(v - maxV)
		exps[i] = e
		sum += e
	}
	invSum := 1.0 / sum
	for i := range exps {
		exps[i] *= invSum
	}
	return exps
}

func categoricalF(probs []float64) int {
	r := rng.Float64()
	csum := 0.0
	for i, p := range probs {
		csum += p
		if r < csum {
			return i
		}
	}
	return len(probs) - 1
}

func gptInference(tokenID, posID int, keys [][][]float64, values [][][]float64, inf inferenceState) []float64 {
	x := vecAddF(inf.wte[tokenID], inf.wpe[posID])
	x = rmsNormF(x)
	for li := range len(inf.layers) {
		layer := inf.layers[li]
		xRes := append([]float64(nil), x...)
		x = rmsNormF(x)
		q := linearF(layer.wq, x)
		k := linearF(layer.wk, x)
		v := linearF(layer.wv, x)
		keys[li] = append(keys[li], k)
		values[li] = append(values[li], v)
		numPast := len(keys[li])
		scale := 1.0 / math.Sqrt(float64(HeadDim))
		xAttn := make([]float64, NEmbd)
		for h := range NHead {
			hs := h * HeadDim
			attnLogits := make([]float64, numPast)
			for t := range numPast {
				keyVec := keys[li][t]
				dot := 0.0
				for j := range HeadDim {
					dot += q[hs+j] * keyVec[hs+j]
				}
				attnLogits[t] = dot * scale
			}
			attnW := softmaxF(attnLogits)
			for j := range HeadDim {
				sumOut := 0.0
				for t := range numPast {
					sumOut += attnW[t] * values[li][t][hs+j]
				}
				xAttn[hs+j] = sumOut
			}
		}
		x = linearF(layer.wo, xAttn)
		x = vecAddF(x, xRes)
		xRes = append([]float64(nil), x...)
		x = rmsNormF(x)
		x = linearF(layer.fc1, x)
		x = reluF(x)
		x = linearF(layer.fc2, x)
		x = vecAddF(x, xRes)
	}
	return linearF(inf.lmHead, x)
}

func computeHeadAttention(headIdx int, q Vec, keysLayer []Vec, valuesLayer []Vec, scaleV *Value) Vec {
	hs := headIdx * HeadDim
	qh := q[hs : hs+HeadDim]
	numPast := len(keysLayer)
	attnLogits := make(Vec, numPast)
	for t := range numPast {
		keyVec := keysLayer[t]
		attnLogits[t] = Mul(Dot(qh, keyVec[hs:hs+HeadDim]), scaleV)
	}
	attnW := Softmax(attnLogits)
	headOut := make(Vec, HeadDim)
	headVectors := make([]Vec, numPast)
	for t := range numPast {
		headVectors[t] = valuesLayer[t][hs : hs+HeadDim]
	}
	for j := range HeadDim {
		headOut[j] = WeightedSum(attnW, headVectors, j)
	}
	return headOut
}

func GPT(tokenID, posID int, keys [][]Vec, values [][]Vec, state map[string]Matrix, layerKeys []layerKeySet) Vec {
	tokEmb := state["wte"][tokenID]
	posEmb := state["wpe"][posID]
	x := VecAdd(tokEmb, posEmb)
	x = RMSNorm(x)
	for li := range NLayer {
		// attention block
		xRes := VecCopy(x)
		x = RMSNorm(x)
		keySet := layerKeys[li]
		var q, k, v Vec
		if NEmbd*NEmbd >= MinParallelQKVWork {
			var qkvWG sync.WaitGroup
			qkvWG.Add(3)
			go func() {
				defer qkvWG.Done()
				q = Linear(state[keySet.wq], x)
			}()
			go func() {
				defer qkvWG.Done()
				k = Linear(state[keySet.wk], x)
			}()
			go func() {
				defer qkvWG.Done()
				v = Linear(state[keySet.wv], x)
			}()
			qkvWG.Wait()
		} else {
			q = Linear(state[keySet.wq], x)
			k = Linear(state[keySet.wk], x)
			v = Linear(state[keySet.wv], x)
		}
		keys[li] = append(keys[li], k)
		values[li] = append(values[li], v)
		numPast := len(keys[li])
		scaleV := NewValue(1.0/math.Sqrt(float64(HeadDim)), nil, nil)
		headOutputs := make([]Vec, NHead)
		if NHead > 1 && numPast*HeadDim >= MinParallelHeadWork {
			headCh := make(chan headResult, NHead)
			for h := range NHead {
				go func(headIdx int) {
					headCh <- headResult{
						idx: headIdx,
						out: computeHeadAttention(headIdx, q, keys[li], values[li], scaleV),
					}
				}(h)
			}
			for range NHead {
				res := <-headCh
				headOutputs[res.idx] = res.out
			}
		} else {
			for h := range NHead {
				headOutputs[h] = computeHeadAttention(h, q, keys[li], values[li], scaleV)
			}
		}
		xAttn := make(Vec, 0, NEmbd)
		for h := range NHead {
			xAttn = append(xAttn, headOutputs[h]...)
		}
		x = Linear(state[keySet.wo], xAttn)
		x = VecAdd(x, xRes)
		// MLP block
		xRes = VecCopy(x)
		x = RMSNorm(x)
		x = Linear(state[keySet.fc1], x)
		x = VecReLU(x)
		x = Linear(state[keySet.fc2], x)
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
	cpuProfilePath := strings.TrimSpace(os.Getenv("MICROGPT_CPU_PROFILE"))
	if cpuProfilePath != "" {
		f, err := os.Create(cpuProfilePath)
		if err != nil {
			fmt.Printf("Error creating CPU profile file: %v\n", err)
			return
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			fmt.Printf("Error starting CPU profile: %v\n", err)
			return
		}
		defer pprof.StopCPUProfile()
	}

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
	layerKeys := make([]layerKeySet, NLayer)
	for i := range NLayer {
		layerKeys[i] = layerKeySet{
			wq:  fmt.Sprintf("layer%d.attn_wq", i),
			wk:  fmt.Sprintf("layer%d.attn_wk", i),
			wv:  fmt.Sprintf("layer%d.attn_wv", i),
			wo:  fmt.Sprintf("layer%d.attn_wo", i),
			fc1: fmt.Sprintf("layer%d.mlp_fc1", i),
			fc2: fmt.Sprintf("layer%d.mlp_fc2", i),
		}
		state[layerKeys[i].wq] = MakeMatrix(NEmbd, NEmbd, std)
		state[layerKeys[i].wk] = MakeMatrix(NEmbd, NEmbd, std)
		state[layerKeys[i].wv] = MakeMatrix(NEmbd, NEmbd, std)
		state[layerKeys[i].wo] = MakeMatrix(NEmbd, NEmbd, std)
		state[layerKeys[i].fc1] = MakeMatrix(4*NEmbd, NEmbd, std)
		state[layerKeys[i].fc2] = MakeMatrix(NEmbd, 4*NEmbd, std)
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
	for step := range NumSteps {
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
		invN := 1.0 / float64(n)
		totalLoss := 0.0
		lossChildren := make([]*Value, 0, n*vocabSize)
		lossLocalGrads := make([]float64, 0, n*vocabSize)
		for pos := range n {
			tokenID := tokens[pos]
			targetID := tokens[pos+1]
			logits := GPT(tokenID, pos, keys, values, state, layerKeys)
			logitsF := make([]float64, len(logits))
			for i := range logits {
				logitsF[i] = logits[i].Data
			}
			probs := softmaxF(logitsF)
			totalLoss += -math.Log(probs[targetID])
			for i := range logits {
				grad := probs[i]
				if i == targetID {
					grad -= 1.0
				}
				lossChildren = append(lossChildren, logits[i])
				lossLocalGrads = append(lossLocalGrads, grad*invN)
			}
		}
		loss := NewValue(totalLoss*invN, lossChildren, lossLocalGrads)
		topo := loss.Backward()
		lossValue := loss.Data
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
		releaseGraph(topo)
		fmt.Printf("step %4d / %4d | loss %.4f\r", step+1, NumSteps, lossValue)
	}
	fmt.Println()
	// inference
	fmt.Println("--- inference (new, hallucinated names) ---")
	temp := 0.5
	inf := buildInferenceState(state, layerKeys)
	for sidx := range 20 {
		keys := make([][][]float64, NLayer)
		values := make([][][]float64, NLayer)
		for i := range keys {
			keys[i] = [][]float64{}
			values[i] = [][]float64{}
		}
		sample := []rune{}
		tokenID := BOS
		for pos := range BlockSize {
			logits := gptInference(tokenID, pos, keys, values, inf)
			invTemp := 1.0 / temp
			tempLogits := make([]float64, len(logits))
			for i, l := range logits {
				tempLogits[i] = l * invTemp
			}
			probs := softmaxF(tempLogits)
			nextToken := categoricalF(probs)
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

	memProfilePath := strings.TrimSpace(os.Getenv("MICROGPT_MEM_PROFILE"))
	if memProfilePath != "" {
		f, err := os.Create(memProfilePath)
		if err != nil {
			fmt.Printf("Error creating memory profile file: %v\n", err)
			return
		}
		defer f.Close()
		runtime.GC()
		if err := pprof.WriteHeapProfile(f); err != nil {
			fmt.Printf("Error writing memory profile: %v\n", err)
			return
		}
	}
}
