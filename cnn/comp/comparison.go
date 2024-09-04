package comp

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"math/big"
	"os"
	"strconv"

	"github.com/lifejade/cnn_ckksv5/cnn/scaleinv"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

func RunMinimaxReLU(compNo, alpha, depthNum, logModulus int, deg []int, scaledVal float64, tree []Tree) {

	// tree: copied tree slice, deg: unchanged

	logQ := []int{60}
	for i := 0; i < depthNum; i++ {
		logQ = append(logQ, logModulus)
	}

	params, err := hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{
		LogN:            16,
		LogQ:            logQ,
		LogP:            []int{60, 60, 60},
		LogDefaultScale: logModulus,
	})
	scaleinv.PrintErr(err)

	// generate classes
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)
	evaluator := hefloat.NewEvaluator(params, evk)
	context := scaleinv.ScaleContext{
		Encoder_:   encoder,
		Encryptor_: encryptor,
		Decryptor_: decryptor,
		Eval_:      evaluator,
		Params_:    params,
	}

	// input vectors
	logn := params.LogN() - 1
	n := 1 << logn
	mx := make([]complex128, n)
	for i := 0; i < n; i++ {
		mx[i] = complex(-1.0+2.0*float64(i)/float64(n-1), 0.0)
	}

	// encryption
	plaintext := hefloat.NewPlaintext(params, params.MaxLevel())
	encoder.Encode(mx, plaintext)
	cipherx, err := encryptor.EncryptNew(plaintext)
	scaleinv.PrintErr(err)

	// evaluation
	fmt.Println("MinimaxReLU")
	scaleinv.DecryptPrint(cipherx, context)

	// minimaxReLU
	cipherx = MinimaxReLU(compNo, alpha, deg, tree, scaledVal, context, cipherx) // deg, tree: unchanged

	scaleinv.DecryptPrint(cipherx, context)
	ShowFailureReLU(cipherx, mx, alpha, context)

}
func MinimaxReLU(compNo, alpha int, deg []int, tree []Tree, finalScaledVal float64, context scaleinv.ScaleContext, cipherIn *rlwe.Ciphertext) (cipherOut *rlwe.Ciphertext) {

	// deg, tree: unchanged
	// classes
	params := context.Params_
	// encoder := context.Encoder_
	// encryptor := context.Encryptor_
	evaluator := context.Eval_

	decompCoeff := make([][]float64, compNo)
	scaledVal := make([]float64, compNo)

	// input file
	addr := "parameters/decomposition"
	str := addr + "/decompcoeff" + strconv.Itoa(alpha) + ".txt"
	file, err := os.Open(str)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	in := bufio.NewScanner(file)

	// scaled value setting
	scaledVal[0] = 1.0
	for i := 1; i < compNo; i++ {
		scaledVal[i] = 2.0
	}
	scaledVal[compNo-1] = finalScaledVal

	// print degrees and coefficients of the component polynomials of minimax composite polynomial
	/* 	for i := 0; i < compNo; i++ {
	   		fmt.Print(deg[i], " ")
	   	}
	   	fmt.Println() */
	for i := 0; i < compNo; i++ {
		for j := 0; j < coeffNumber(deg[i], tree[i]); j++ {
			var buffer []float64
			buffer = ReadLines(in, buffer, 1)
			decompCoeff[i] = append(decompCoeff[i], buffer[0])
			// fmt.Print(decompCoeff[i][j], " ")
		}
		// fmt.Println()
	}

	// generation of half ciphertext
	logn := params.LogN() - 1
	n := 1 << logn
	mHalf := make([]float64, n)
	for i := 0; i < n; i++ {
		mHalf[i] = 0.5
	}
	cipherX := cipherIn.CopyNew()

	// evaluating pk ... p1(x) / 2
	for i := 0; i < compNo; i++ {
		// fmt.Println("*******************************************")
		// fmt.Println("               No: ", i)
		cipherX = evalPolynomialIntegrate(cipherX, deg[i], tree[i], decompCoeff[i], context) // decompCoeff: unchanged
		scaleinv.DecryptPrint(cipherX, context)
	}

	// x(1+sgn(x))/2 from sgn(x)/2

	temp1 := cipherX.CopyNew()
	err = evaluator.Add(temp1, 0.5, temp1)
	scaleinv.PrintErr(err)

	lprime := temp1.Level()
	Delta := math.Pow(2, temp1.LogScale())

	tempint := new(big.Int)
	tempint.Div(params.RingQ().ModulusAtLevel[lprime], params.RingQ().ModulusAtLevel[lprime-1])
	tempfloat := new(big.Float).SetInt(tempint)
	tmp, _ := tempfloat.Float64()

	Delta_prime := math.Pow(2, cipherIn.LogScale()) / Delta * tmp

	temp2 := cipherIn.CopyNew()
	evaluator.SetScale(temp2, rlwe.NewScale(Delta_prime))
	evaluator.DropLevel(temp2, temp2.Level()-lprime)
	err = evaluator.MulRelin(temp1, temp2, temp1)
	scaleinv.PrintErr(err)
	evaluator.Rescale(temp1, temp1)
	cipherOut = temp1.CopyNew()

	return cipherOut

}

func evalPolynomialIntegrate(cipher *rlwe.Ciphertext, deg int, tree Tree, decompCoeff []float64, context scaleinv.ScaleContext) (res *rlwe.Ciphertext) {

	// decompCoeff: unchanged
	// classes
	// params := context.Params_
	// encoder := context.Encoder_
	// encryptor := context.Encryptor_
	evaluator := context.Eval_

	// scale := cipher.Scale
	// n := context.Params_.LogSlots()
	totalDepth := int(math.Ceil(math.Log2(float64(deg + 1))))

	decompDeg := make([]int, Pow2(tree.depth+1))
	startIndex := make([]int, Pow2(tree.depth+1))
	for i := 0; i < Pow2(tree.depth+1); i++ {
		decompDeg[i] = -1
		startIndex[i] = -1
	}

	T := make([]*rlwe.Ciphertext, 100)
	pt := make([]*rlwe.Ciphertext, 100)

	// set start temp_index
	// num := 0
	tempIndex := 1

	// evaluate decompose polynomial degrees
	decompDeg[1] = deg
	for i := 1; i <= tree.depth; i++ {
		for j := Pow2(i); j < Pow2(i+1); j++ {

			if j >= len(decompDeg) {
				fmt.Println("invalid index")
				os.Exit(1)
			}
			if j%2 == 0 {
				decompDeg[j] = tree.tree[j/2] - 1
			} else if j%2 == 1 {
				decompDeg[j] = decompDeg[j/2] - tree.tree[j/2]
			}
		}
	}

	// compute start index
	for i := 1; i < Pow2(tree.depth+1); i++ {
		if tree.tree[i] == 0 {
			startIndex[i] = tempIndex
			tempIndex += (decompDeg[i] + 1)
		}
	}

	// generate T0, T1
	T[0], T[1] = geneT0T1(cipher, context)

	// fmt.Println("T: ", 0)
	// scaleinv.DecryptPrint(T[0], context)
	// fmt.Println("T: ", 1)
	// scaleinv.DecryptPrint(T[1], context)

	// i: depth stage
	for i := 1; i <= totalDepth; i++ {

		// fmt.Println("////////////// stage : ", i)

		for j := 1; j < Pow2(tree.depth+1); j++ {
			if tree.tree[j] == 0 && totalDepth+1-NumOne(j) == i {

				tempIdx := startIndex[j]
				// pt[j] = scaleinv.MultByConstNew(T[1], decompCoeff[tempIdx], context)
				// scaleinv.MultByConstDouble(T[1], pt[j], decompCoeff[tempIdx], context)
				pt[j] = scaleinv.MultByConstDoubleNew(T[1], decompCoeff[tempIdx], context)
				tempIdx += 2
				for k := 3; k <= decompDeg[j]; k += 2 {
					// temp1 := scaleinv.MultByConstNew(T[k], decompCoeff[tempIdx], context)
					temp1 := scaleinv.MultByConstDoubleNew(T[k], decompCoeff[tempIdx], context)
					scaleinv.AddScaleInv(pt[j], temp1, pt[j], context)
					tempIdx += 2
				}
				// evaluator.Rescale(pt[j], params.DefaultScale(), pt[j])
				evaluator.Rescale(pt[j], pt[j])

				// fmt.Println("pt: ", j)
				// scaleinv.DecryptPrint(pt[j], context)

			}
		}

		// depth i computation. all intersection points.

		for j := 1; j < Pow2(tree.depth+1); j++ {
			if tree.tree[j] > 0 && totalDepth+1-NumOne(j) == i && j%2 == 1 {

				k := j
				pt[j] = scaleinv.MultScaleInvNew(T[tree.tree[k]], pt[2*k+1], context)
				k *= 2
				for {
					if tree.tree[k] == 0 {
						break
					}
					temp1 := scaleinv.MultScaleInvNew(T[tree.tree[k]], pt[2*k+1], context)
					scaleinv.AddScaleInv(pt[j], temp1, pt[j], context)
					k *= 2
				}
				// evaluator.Rescale(pt[j], params.DefaultScale(), pt[j])
				err := evaluator.Rescale(pt[j], pt[j])
				scaleinv.PrintErr(err)
				scaleinv.AddScaleInv(pt[j], pt[k], pt[j], context)

				// fmt.Println("pt: ", j)
				// scaleinv.DecryptPrint(pt[j], context)

			}
		}

		// Ti evaluation
		if i <= tree.m-1 {

			T[Pow2(i)] = evalT(T[Pow2(i-1)], T[Pow2(i-1)], T[0], context)

			// fmt.Println("T: ", Pow2(i))
			// scaleinv.DecryptPrint(T[Pow2(i)], context)
		}

		if i <= tree.l {

			for j := Pow2(i-1) + 1; j <= Pow2(i)-1; j += 2 {
				T[j] = evalT(T[Pow2(i-1)], T[j-Pow2(i-1)], T[Pow2(i)-j], context)

				// fmt.Println("T: ", j)
				// scaleinv.DecryptPrint(T[j], context)
			}
		}

	}

	res = pt[1].CopyNew()
	return res

}

func evalT(Tm, Tn, Tmminusn *rlwe.Ciphertext, context scaleinv.ScaleContext) (Tmplusn *rlwe.Ciphertext) {

	evaluator := context.Eval_

	temp := scaleinv.MultScaleInvNew(Tm, Tn, context)
	// scaleinv.AddScaleInv(temp, temp, temp, context)
	temp = scaleinv.AddScaleInvNew(temp, temp, context)
	err := evaluator.Rescale(temp, temp)
	scaleinv.PrintErr(err)
	Tmplusn = scaleinv.SubScaleInvNew(temp, Tmminusn, context)

	return Tmplusn
}

func geneT0T1(cipher *rlwe.Ciphertext, context scaleinv.ScaleContext) (T0, T1 *rlwe.Ciphertext) {

	params := context.Params_
	encoder := context.Encoder_
	encryptor := context.Encryptor_

	// scale := cipher.Scale
	// logn := params.LogSlots()
	// n := params.Slots()
	logn := params.LogN() - 1
	n := 1 << logn
	mOne := make([]float64, n)

	// ctxt_1
	for i := 0; i < n; i++ {
		mOne[i] = 1.0
	}

	plain1 := hefloat.NewPlaintext(params, params.MaxLevel())
	plain1.Scale = cipher.Scale
	err := encoder.Encode(mOne, plain1)
	scaleinv.PrintErr(err)
	ctxt1, err := encryptor.EncryptNew(plain1)
	scaleinv.PrintErr(err)
	// plain1 := ckks.NewPlaintext(params, params.MaxLevel(), scale)
	// encoder.Encode(mOne, plain1, logn)
	// ctxt1 := encryptor.EncryptNew(plain1)

	T0 = ctxt1
	T1 = cipher.CopyNew()

	return T0, T1

}
func coeffNumber(deg int, tree Tree) (num int) {
	// tree: unchanged

	num = 0
	decompDeg := make([]int, Pow2(tree.depth+1))
	decompDeg[1] = deg

	for i := 1; i <= tree.depth; i++ {
		for j := Pow2(i); j < Pow2(i+1); j++ {

			if j%2 == 0 {
				decompDeg[j] = tree.tree[j/2] - 1
			} else if j%2 == 1 {
				decompDeg[j] = decompDeg[j/2] - tree.tree[j/2]
			}
		}
	}

	for i := 0; i < Pow2(tree.depth+1); i++ {
		if tree.tree[i] == 0 {
			num += (decompDeg[i] + 1)
		}
	}

	return num

}
