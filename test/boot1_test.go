package test

import (
	"fmt"
	"math/big"
	"runtime"
	"testing"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"
	"github.com/tuneinsight/lattigo/v5/ring"
	"github.com/tuneinsight/lattigo/v5/utils"
)

// boot params test
func Test_bootparams1(t *testing.T) {
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	//ckksParams := bootstrapping.N16QP1793H32768H32
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            16,
		LogQ:            []int{55, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 30},
		LogP:            []int{61, 61, 61, 61, 61},
		Xs:              ring.Ternary{H: 32768},
		LogDefaultScale: 30,
	}
	bootstrappingParams := bootstrapping.ParametersLiteral{
		SlotsToCoeffsFactorizationDepthAndLogScales: [][]int{{30}, {30, 30}},
		CoeffsToSlotsFactorizationDepthAndLogScales: [][]int{{53}, {53}, {53}, {53}},
		EvalModLogScale: utils.Pointy(55),
	}

	//parameter init
	params, err := hefloat.NewParametersFromLiteral(SchemeParams)
	if err != nil {
		panic(err)
	}
	btpParams, err := bootstrapping.NewParametersFromLiteral(params, bootstrappingParams)
	if err != nil {
		panic(err)
	}
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

	fmt.Println("make boot params")
	btpevk, _, _ := btpParams.GenEvaluationKeys(sk)

	btp, _ := bootstrapping.NewEvaluator(btpParams, btpevk)

	_, _, _, _, _ = encryptor, decryptor, encoder, evaluator, btp

	fmt.Println("end boot params")

	n := 1 << params.LogMaxSlots()
	value := make([]complex128, n)
	for i := range value {
		if i < 100 {
			value[i] = complex(float64(i), 0)
		} else {
			value[i] = complex(0, 0)
		}
	}
	plaintext := hefloat.NewPlaintext(params, params.MaxLevel())
	encoder.Encode(value, plaintext)
	cipher, _ := encryptor.EncryptNew(plaintext)

	DecryptPrint(params, cipher, *decryptor, *encoder)

	res, _ := btp.Bootstrap(cipher)

	DecryptPrint(params, res, *decryptor, *encoder)
}

func Test_bootparams2(t *testing.T) {
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            16,
		LogQ:            []int{60, 45, 45, 45, 45, 45, 42, 42},
		LogP:            []int{61, 61, 61, 61},
		Xs:              ring.Ternary{H: 192},
		LogDefaultScale: 45,
	}
	bootstrappingParams := bootstrapping.ParametersLiteral{
		SlotsToCoeffsFactorizationDepthAndLogScales: [][]int{{42}, {42}, {42}},
		CoeffsToSlotsFactorizationDepthAndLogScales: [][]int{{58}, {58}, {58}, {58}},
		LogMessageRatio: utils.Pointy(2),
		Mod1InvDegree:   utils.Pointy(7),
	}

	//parameter init
	params, err := hefloat.NewParametersFromLiteral(SchemeParams)
	if err != nil {
		panic(err)
	}
	btpParams, err := bootstrapping.NewParametersFromLiteral(params, bootstrappingParams)
	if err != nil {
		panic(err)
	}
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

	fmt.Println("make boot params")
	btpevk, _, _ := btpParams.GenEvaluationKeys(sk)

	btp, _ := bootstrapping.NewEvaluator(btpParams, btpevk)

	_, _, _, _, _ = encryptor, decryptor, encoder, evaluator, btp

	fmt.Println("end boot params")

	n := 1 << params.LogMaxSlots()
	value := make([]complex128, n)
	for i := range value {
		if i < 100 {
			value[i] = complex(float64(i), 0)
		} else {
			value[i] = complex(0, 0)
		}
	}
	plaintext := hefloat.NewPlaintext(params, params.MaxLevel())
	encoder.Encode(value, plaintext)
	cipher, _ := encryptor.EncryptNew(plaintext)

	DecryptPrint(params, cipher, *decryptor, *encoder)

	res, _ := btp.Bootstrap(cipher)

	DecryptPrint(params, res, *decryptor, *encoder)
}

func Test_bootparams3(t *testing.T) {
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            16,
		LogQ:            []int{60, 45, 45, 45, 45, 45, 42, 42},
		LogP:            []int{61, 61, 61, 61},
		Xs:              ring.Ternary{H: 192},
		LogDefaultScale: 45,
	}
	bootstrappingParams := bootstrapping.ParametersLiteral{
		SlotsToCoeffsFactorizationDepthAndLogScales: [][]int{{42}, {42}, {42}},
		CoeffsToSlotsFactorizationDepthAndLogScales: [][]int{{58}, {58}, {58}, {58}},
		LogMessageRatio: utils.Pointy(2),
		Mod1InvDegree:   utils.Pointy(7),
	}

	//parameter init
	params, err := hefloat.NewParametersFromLiteral(SchemeParams)
	if err != nil {
		panic(err)
	}
	btpParams, err := bootstrapping.NewParametersFromLiteral(params, bootstrappingParams)
	if err != nil {
		panic(err)
	}
	btpParams.SlotsToCoeffsParameters.Scaling = new(big.Float).SetFloat64(0.5)
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

	fmt.Println("make boot params")
	btpevk, _, _ := btpParams.GenEvaluationKeys(sk)

	btp, _ := bootstrapping.NewEvaluator(btpParams, btpevk)

	_, _, _, _, _ = encryptor, decryptor, encoder, evaluator, btp

	fmt.Println("end boot params")

	n := 1 << params.LogMaxSlots()
	value := make([]complex128, n)
	for i := range value {
		if i < 100 {
			value[i] = complex(float64(i), 0)
		} else {
			value[i] = complex(0, 0)
		}
	}
	plaintext := hefloat.NewPlaintext(params, params.MaxLevel())
	encoder.Encode(value, plaintext)
	cipher, _ := encryptor.EncryptNew(plaintext)

	DecryptPrint(params, cipher, *decryptor, *encoder)

	res, _ := btp.Bootstrap(cipher)

	DecryptPrint(params, res, *decryptor, *encoder)
}

func DecryptPrint(params hefloat.Parameters, ciphertext *rlwe.Ciphertext, decryptor rlwe.Decryptor, encoder hefloat.Encoder) {

	N := 1 << params.LogN()
	n := N / 2
	message := make([]complex128, n)
	encoder.Decode(decryptor.DecryptNew(ciphertext), message)

	fmt.Println()
	fmt.Printf("Level: %d (logQ = %d)\n", ciphertext.Level(), params.LogQLvl(ciphertext.Level()))
	fmt.Printf("Scale: 2^%f\n", ciphertext.LogScale())
	fmt.Printf("Values: %6.10f %6.10f %6.10f %6.10f %6.10f...\n", message[0], message[1], message[2], message[3], message[4])

	// max, min
	max, min := 0.0, 1.0
	for _, v := range message {
		if max < real(v) {
			max = real(v)
		}
		if min > real(v) {
			min = real(v)
		}
	}

	fmt.Println("Max, Min value: ", max, " ", min)
	fmt.Println()

	return

}
