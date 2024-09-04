package scaleinv

import (
	"fmt"
	"math"
	"math/big"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

type ScaleContext struct {
	Encoder_   *hefloat.Encoder
	Encryptor_ *rlwe.Encryptor
	Decryptor_ *rlwe.Decryptor
	Eval_      *hefloat.Evaluator
	Params_    hefloat.Parameters
}

func DecryptPrint(ciphertext *rlwe.Ciphertext, context ScaleContext) {

	params := context.Params_
	encoder := context.Encoder_
	decryptor := context.Decryptor_

	N := 1 << params.LogN()
	n := N / 2

	message := make([]complex128, n)
	encoder.Decode(decryptor.DecryptNew(ciphertext), message)

	fmt.Println()
	fmt.Printf("Level: %d (logQ = %d)\n", ciphertext.Level(), params.LogQLvl(ciphertext.Level()))
	fmt.Printf("Scale: 2^%f\n", ciphertext.LogScale())
	fmt.Printf("Values: %6.10f %6.10f %6.10f %6.10f %6.10f...%6.10f\n", message[0], message[1], message[2], message[3], message[4], message[n-1])

	// max, min real part
	max, min := -100.0, 100.0
	for _, v := range message {
		if max < real(v) {
			max = real(v)
		}
		if min > real(v) {
			min = real(v)
		}
	}

	// maxabs := 0.0
	// for _, v := range message {
	// 	if cmplx.Abs(v) > maxabs {
	// 		maxabs = cmplx.Abs(v)
	// 	}
	// }

	fmt.Println("Min, Max value (real part): ", min, " ", max)
	// fmt.Println("Max abs value (complex): ", maxabs)
	fmt.Println()

}

func MultByConstDoubleNew(ctxtIn *rlwe.Ciphertext, constVal float64, context ScaleContext) (ctxtOut *rlwe.Ciphertext) {

	params := context.Params_
	encoder := context.Encoder_
	evaluator := context.Eval_
	logn := params.LogN() - 1
	n := 1 << logn

	constVec := make([]complex128, n)
	for i := 0; i < n; i++ {
		constVec[i] = complex(constVal, 0.0)
	}
	plain := hefloat.NewPlaintext(params, ctxtIn.Level())
	plain.Scale = ctxtIn.Scale
	encoder.Encode(constVec, plain)
	ctxtOut, _ = evaluator.MulNew(ctxtIn, plain)

	return ctxtOut
}
func MultByConstDouble(ctxtIn *rlwe.Ciphertext, ctxtOut *rlwe.Ciphertext, constVal float64, context ScaleContext) {

	params := context.Params_
	encoder := context.Encoder_
	evaluator := context.Eval_
	logn := params.LogN() - 1
	n := 1 << logn

	constVec := make([]complex128, n)
	for i := 0; i < n; i++ {
		constVec[i] = complex(constVal, 0.0)
	}
	plain := hefloat.NewPlaintext(params, ctxtIn.Level())
	plain.Scale = ctxtIn.Scale
	encoder.Encode(constVec, plain)
	err := evaluator.Mul(ctxtIn, plain, ctxtOut)
	PrintErr(err)

}

func MultByConstVecTarget(cipher *rlwe.Ciphertext, constVec interface{}, context ScaleContext, targetScale float64) (res *rlwe.Ciphertext) {

	params := context.Params_
	encoder := context.Encoder_
	evaluator := context.Eval_

	cipher_scale := math.Pow(2, cipher.LogScale())
	level := cipher.Level()
	tempint := new(big.Int)
	tempint.Div(params.RingQ().ModulusAtLevel[level], params.RingQ().ModulusAtLevel[level-1])
	tempfloat := new(big.Float).SetInt(tempint)
	tmp, _ := tempfloat.Float64()
	scale := tmp * targetScale / cipher_scale

	plain := hefloat.NewPlaintext(params, cipher.Level())
	plain.Scale = rlwe.NewScale(scale)
	encoder.Encode(constVec, plain)
	res, _ = evaluator.MulNew(cipher, plain)

	return res
}
func MultByConstTarget(cipher *rlwe.Ciphertext, constant float64, context ScaleContext, targetScale float64) (res *rlwe.Ciphertext) {

	params := context.Params_
	encoder := context.Encoder_
	evaluator := context.Eval_
	logn := params.LogN() - 1
	n := 1 << logn

	cipher_scale := math.Pow(2, cipher.LogScale())
	level := cipher.Level()
	tempint := new(big.Int)
	tempint.Div(params.RingQ().ModulusAtLevel[level], params.RingQ().ModulusAtLevel[level-1])
	tempfloat := new(big.Float).SetInt(tempint)
	tmp, _ := tempfloat.Float64()
	scale := tmp * targetScale / cipher_scale

	plain := hefloat.NewPlaintext(params, cipher.Level())
	plain.Scale = rlwe.NewScale(scale)
	constVec := make([]float64, n)
	for i := range constVec {
		constVec[i] = constant
	}

	encoder.Encode(constVec, plain)
	res, _ = evaluator.MulNew(cipher, plain)

	return res
}

func AddScaleInv(cipher0, cipher1, res *rlwe.Ciphertext, context ScaleContext) {

	params := context.Params_
	encoder := context.Encoder_
	evaluator := context.Eval_
	logn := params.LogN() - 1
	n := 1 << logn

	if cipher0.Level() < cipher1.Level() {

		scale0 := math.Pow(2, cipher0.LogScale())
		scale1 := math.Pow(2, cipher1.LogScale())

		level1 := cipher1.Level()
		tempint := new(big.Int)
		tempint.Div(params.RingQ().ModulusAtLevel[level1], params.RingQ().ModulusAtLevel[level1-1])

		tempfloat := new(big.Float).SetInt(tempint)
		tmp, _ := tempfloat.Float64()
		scale := scale0 / scale1 * tmp
		values1 := make([]complex128, n)
		for i := range values1 {
			values1[i] = 1.0
		}

		// scaler
		scaler := hefloat.NewPlaintext(params, cipher1.Level())
		scaler.Scale = rlwe.NewScale(scale)
		encoder.Encode(values1, scaler)
		temp, _ := evaluator.MulRelinNew(cipher1, scaler)

		evaluator.Rescale(temp, temp)
		evaluator.DropLevel(temp, cipher1.Level()-cipher0.Level()-1)
		evaluator.Add(cipher0, temp, res)

	} else if cipher0.Level() > cipher1.Level() {

		scale0 := math.Pow(2, cipher0.LogScale())
		scale1 := math.Pow(2, cipher1.LogScale())
		level0 := cipher0.Level()
		tempint := new(big.Int)
		tempint.Div(params.RingQ().ModulusAtLevel[level0], params.RingQ().ModulusAtLevel[level0-1])

		tempfloat := new(big.Float).SetInt(tempint)
		tmp, _ := tempfloat.Float64()

		scale := scale1 / scale0 * tmp
		values0 := make([]complex128, n)
		for i := range values0 {
			values0[i] = 1.0
		}

		// scaler
		scaler := hefloat.NewPlaintext(params, cipher0.Level())
		scaler.Scale = rlwe.NewScale(scale)
		encoder.Encode(values0, scaler)
		temp, _ := evaluator.MulRelinNew(cipher0, scaler)

		evaluator.Rescale(temp, temp)
		evaluator.DropLevel(temp, cipher0.Level()-cipher1.Level()-1)
		evaluator.Add(cipher1, temp, res)

	} else {
		evaluator.Add(cipher0, cipher1, res)
	}

}
func AddScaleInvNew(cipher0, cipher1 *rlwe.Ciphertext, context ScaleContext) (res *rlwe.Ciphertext) {

	params := context.Params_
	encoder := context.Encoder_
	evaluator := context.Eval_
	logn := params.LogN() - 1
	n := 1 << logn

	if cipher0.Level() < cipher1.Level() {

		scale0 := math.Pow(2, cipher0.LogScale())
		scale1 := math.Pow(2, cipher1.LogScale())

		level1 := cipher1.Level()
		tempint := new(big.Int)
		tempint.Div(params.RingQ().ModulusAtLevel[level1], params.RingQ().ModulusAtLevel[level1-1])

		tempfloat := new(big.Float).SetInt(tempint)
		tmp, _ := tempfloat.Float64()
		scale := scale0 / scale1 * tmp
		values1 := make([]complex128, n)
		for i := range values1 {
			values1[i] = 1.0
		}

		// scaler
		scaler := hefloat.NewPlaintext(params, cipher1.Level())
		scaler.Scale = rlwe.NewScale(scale)
		encoder.Encode(values1, scaler)
		temp, _ := evaluator.MulRelinNew(cipher1, scaler)

		evaluator.Rescale(temp, temp)
		evaluator.DropLevel(temp, cipher1.Level()-cipher0.Level()-1)
		res, _ = evaluator.AddNew(cipher0, temp)

	} else if cipher0.Level() > cipher1.Level() {

		scale0 := math.Pow(2, cipher0.LogScale())
		scale1 := math.Pow(2, cipher1.LogScale())
		level0 := cipher0.Level()
		tempint := new(big.Int)
		tempint.Div(params.RingQ().ModulusAtLevel[level0], params.RingQ().ModulusAtLevel[level0-1])

		tempfloat := new(big.Float).SetInt(tempint)
		tmp, _ := tempfloat.Float64()

		scale := scale1 / scale0 * tmp
		values0 := make([]complex128, n)
		for i := range values0 {
			values0[i] = 1.0
		}

		// scaler
		scaler := hefloat.NewPlaintext(params, cipher0.Level())
		scaler.Scale = rlwe.NewScale(scale)
		encoder.Encode(values0, scaler)
		temp, _ := evaluator.MulRelinNew(cipher0, scaler)

		evaluator.Rescale(temp, temp)
		evaluator.DropLevel(temp, cipher0.Level()-cipher1.Level()-1)
		res, _ = evaluator.AddNew(cipher1, temp)

	} else {
		res, _ = evaluator.AddNew(cipher0, cipher1)
	}

	return res

}
func SubScaleInv(cipher0, cipher1, res *rlwe.Ciphertext, context ScaleContext) {

	params := context.Params_
	encoder := context.Encoder_
	evaluator := context.Eval_
	encryptor := context.Encryptor_
	logn := params.LogN() - 1
	n := 1 << logn

	// cipher1 = -cipher1
	temp := cipher1.CopyNew()
	err := encryptor.EncryptZero(temp)
	PrintErr(err)
	err = evaluator.Sub(temp, cipher1, cipher1)
	PrintErr(err)

	if cipher0.Level() < cipher1.Level() {

		scale0 := math.Pow(2, cipher0.LogScale())
		scale1 := math.Pow(2, cipher1.LogScale())

		level1 := cipher1.Level()
		tempint := new(big.Int)
		tempint.Div(params.RingQ().ModulusAtLevel[level1], params.RingQ().ModulusAtLevel[level1-1])

		tempfloat := new(big.Float).SetInt(tempint)
		tmp, _ := tempfloat.Float64()
		scale := scale0 / scale1 * tmp
		values1 := make([]complex128, n)
		for i := range values1 {
			values1[i] = 1.0
		}

		// scaler
		scaler := hefloat.NewPlaintext(params, cipher1.Level())
		scaler.Scale = rlwe.NewScale(scale)
		encoder.Encode(values1, scaler)
		temp, _ := evaluator.MulRelinNew(cipher1, scaler)

		evaluator.Rescale(temp, temp)
		evaluator.DropLevel(temp, cipher1.Level()-cipher0.Level()-1)
		evaluator.Add(cipher0, temp, res)

	} else if cipher0.Level() > cipher1.Level() {

		scale0 := math.Pow(2, cipher0.LogScale())
		scale1 := math.Pow(2, cipher1.LogScale())
		level0 := cipher0.Level()
		tempint := new(big.Int)
		tempint.Div(params.RingQ().ModulusAtLevel[level0], params.RingQ().ModulusAtLevel[level0-1])

		tempfloat := new(big.Float).SetInt(tempint)
		tmp, _ := tempfloat.Float64()

		scale := scale1 / scale0 * tmp
		values0 := make([]complex128, n)
		for i := range values0 {
			values0[i] = 1.0
		}

		// scaler
		scaler := hefloat.NewPlaintext(params, cipher0.Level())
		scaler.Scale = rlwe.NewScale(scale)
		encoder.Encode(values0, scaler)
		temp, _ := evaluator.MulRelinNew(cipher0, scaler)

		evaluator.Rescale(temp, temp)
		evaluator.DropLevel(temp, cipher0.Level()-cipher1.Level()-1)
		evaluator.Add(cipher1, temp, res)

	} else {
		evaluator.Add(cipher0, cipher1, res)
	}

}
func SubScaleInvNew(cipher0, cipher1 *rlwe.Ciphertext, context ScaleContext) (res *rlwe.Ciphertext) {

	params := context.Params_
	encoder := context.Encoder_
	evaluator := context.Eval_
	encryptor := context.Encryptor_
	logn := params.LogN() - 1
	n := 1 << logn

	// cipher1 = -cipher1
	temp := cipher1.CopyNew()
	err := encryptor.EncryptZero(temp)
	PrintErr(err)
	cipher1_neg, err := evaluator.SubNew(temp, cipher1)
	PrintErr(err)

	if cipher0.Level() < cipher1_neg.Level() {

		scale0 := math.Pow(2, cipher0.LogScale())
		scale1 := math.Pow(2, cipher1_neg.LogScale())

		level1 := cipher1_neg.Level()
		tempint := new(big.Int)
		tempint.Div(params.RingQ().ModulusAtLevel[level1], params.RingQ().ModulusAtLevel[level1-1])
		tempfloat := new(big.Float).SetInt(tempint)
		tmp, _ := tempfloat.Float64()
		scale := scale0 / scale1 * tmp
		values1 := make([]complex128, n)
		for i := range values1 {
			values1[i] = 1.0
		}

		// scaler
		scaler := hefloat.NewPlaintext(params, cipher1_neg.Level())
		scaler.Scale = rlwe.NewScale(scale)
		encoder.Encode(values1, scaler)
		temp, _ := evaluator.MulRelinNew(cipher1_neg, scaler)

		evaluator.Rescale(temp, temp)
		evaluator.DropLevel(temp, cipher1_neg.Level()-cipher0.Level()-1)
		res, _ = evaluator.AddNew(cipher0, temp)

	} else if cipher0.Level() > cipher1_neg.Level() {

		scale0 := math.Pow(2, cipher0.LogScale())
		scale1 := math.Pow(2, cipher1_neg.LogScale())
		level0 := cipher0.Level()
		tempint := new(big.Int)
		tempint.Div(params.RingQ().ModulusAtLevel[level0], params.RingQ().ModulusAtLevel[level0-1])

		tempfloat := new(big.Float).SetInt(tempint)
		tmp, _ := tempfloat.Float64()
		scale := scale1 / scale0 * tmp
		values0 := make([]complex128, n)
		for i := range values0 {
			values0[i] = 1.0
		}

		// scaler
		scaler := hefloat.NewPlaintext(params, cipher0.Level())
		scaler.Scale = rlwe.NewScale(scale)
		encoder.Encode(values0, scaler)
		temp, _ := evaluator.MulRelinNew(cipher0, scaler)

		evaluator.Rescale(temp, temp)
		evaluator.DropLevel(temp, cipher0.Level()-cipher1_neg.Level()-1)
		res, _ = evaluator.AddNew(cipher1_neg, temp)

	} else {
		res, _ = evaluator.AddNew(cipher0, cipher1_neg)
	}

	return res

}
func MultScaleInv(cipher0, cipher1, res *rlwe.Ciphertext, context ScaleContext) {

	params := context.Params_
	encoder := context.Encoder_
	evaluator := context.Eval_
	logn := params.LogN() - 1
	n := 1 << logn

	if cipher0.Level() < cipher1.Level() {

		scale0 := math.Pow(2, cipher0.LogScale())
		scale1 := math.Pow(2, cipher1.LogScale())

		level1 := cipher1.Level()
		tempint := new(big.Int)
		tempint.Div(params.RingQ().ModulusAtLevel[level1], params.RingQ().ModulusAtLevel[level1-1])

		tempfloat := new(big.Float).SetInt(tempint)
		tmp, _ := tempfloat.Float64()
		scale := scale0 / scale1 * tmp
		values1 := make([]complex128, n)
		for i := range values1 {
			values1[i] = 1.0
		}

		// scalar
		scaler := hefloat.NewPlaintext(params, cipher1.Level())
		scaler.Scale = rlwe.NewScale(scale)
		encoder.Encode(values1, scaler)
		temp, _ := evaluator.MulRelinNew(cipher1, scaler)

		evaluator.Rescale(temp, temp)
		evaluator.DropLevel(temp, cipher1.Level()-cipher0.Level()-1)
		evaluator.MulRelin(cipher0, temp, res)

	} else if cipher0.Level() > cipher1.Level() {

		scale0 := math.Pow(2, cipher0.LogScale())
		scale1 := math.Pow(2, cipher1.LogScale())

		level0 := cipher0.Level()
		tempint := new(big.Int)
		tempint.Div(params.RingQ().ModulusAtLevel[level0], params.RingQ().ModulusAtLevel[level0-1])

		tempfloat := new(big.Float).SetInt(tempint)
		tmp, _ := tempfloat.Float64()
		scale := scale1 / scale0 * tmp
		values0 := make([]complex128, n)
		for i := range values0 {
			values0[i] = 1.0
		}

		// scalar
		scaler := hefloat.NewPlaintext(params, cipher0.Level())
		scaler.Scale = rlwe.NewScale(scale)
		encoder.Encode(values0, scaler)
		temp, _ := evaluator.MulRelinNew(cipher0, scaler)

		evaluator.Rescale(temp, temp)
		evaluator.DropLevel(temp, cipher0.Level()-cipher1.Level()-1)
		evaluator.MulRelin(cipher1, temp, res)

	} else {
		evaluator.MulRelin(cipher0, cipher1, res)
	}

}
func MultScaleInvNew(cipher0, cipher1 *rlwe.Ciphertext, context ScaleContext) (res *rlwe.Ciphertext) {

	params := context.Params_
	encoder := context.Encoder_
	evaluator := context.Eval_
	logn := params.LogN() - 1
	n := 1 << logn

	if cipher0.Level() < cipher1.Level() {

		scale0 := math.Pow(2, cipher0.LogScale())
		scale1 := math.Pow(2, cipher1.LogScale())

		level1 := cipher1.Level()
		tempint := new(big.Int)
		tempint.Div(params.RingQ().ModulusAtLevel[level1], params.RingQ().ModulusAtLevel[level1-1])

		tempfloat := new(big.Float).SetInt(tempint)
		tmp, _ := tempfloat.Float64()
		scale := scale0 / scale1 * tmp
		values1 := make([]complex128, n)
		for i := range values1 {
			values1[i] = 1.0
		}

		// scalar
		scaler := hefloat.NewPlaintext(params, cipher1.Level())
		scaler.Scale = rlwe.NewScale(scale)
		encoder.Encode(values1, scaler)
		temp, _ := evaluator.MulRelinNew(cipher1, scaler)

		evaluator.Rescale(temp, temp)
		evaluator.DropLevel(temp, cipher1.Level()-cipher0.Level()-1)
		res, _ = evaluator.MulRelinNew(cipher0, temp)

	} else if cipher0.Level() > cipher1.Level() {

		scale0 := math.Pow(2, cipher0.LogScale())
		scale1 := math.Pow(2, cipher1.LogScale())

		level0 := cipher0.Level()
		tempint := new(big.Int)
		tempint.Div(params.RingQ().ModulusAtLevel[level0], params.RingQ().ModulusAtLevel[level0-1])

		tempfloat := new(big.Float).SetInt(tempint)
		tmp, _ := tempfloat.Float64()
		scale := scale1 / scale0 * tmp
		values0 := make([]complex128, n)
		for i := range values0 {
			values0[i] = 1.0
		}

		// scalar
		scaler := hefloat.NewPlaintext(params, cipher0.Level())
		scaler.Scale = rlwe.NewScale(scale)
		encoder.Encode(values0, scaler)
		temp, _ := evaluator.MulRelinNew(cipher0, scaler)

		evaluator.Rescale(temp, temp)
		evaluator.DropLevel(temp, cipher0.Level()-cipher1.Level()-1)
		res, _ = evaluator.MulRelinNew(cipher1, temp)

	} else {
		res, _ = evaluator.MulRelinNew(cipher0, cipher1)
	}

	return res

}
func PrintErr(err error) {
	if err != nil {
		fmt.Println(err)
	}
}
