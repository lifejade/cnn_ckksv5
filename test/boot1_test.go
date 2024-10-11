package test

import (
	"fmt"
	"math/big"
	"runtime"
	"testing"
	"time"

	"github.com/lifejade/cnn_ckksv5/cnn"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"
	"github.com/tuneinsight/lattigo/v5/ring"
	"github.com/tuneinsight/lattigo/v5/utils"
	"github.com/tuneinsight/lattigo/v5/utils/sampling"
)

// boot params test
func Test_bootparams1(t *testing.T) {
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	//ckksParams := bootstrapping.N16QP1793H32768H32
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            16,
		LogQ:            []int{55, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60},
		LogP:            []int{61, 61, 61, 61, 61},
		Xs:              ring.Ternary{H: 32768},
		LogDefaultScale: 30,
	}
	bootstrappingParams := bootstrapping.ParametersLiteral{
		SlotsToCoeffsFactorizationDepthAndLogScales: [][]int{{30, 30}, {30, 30}},
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

	fmt.Printf("%d  %d  %d\n", btp.Depth(), btp.OutputLevel(), btp.MinimumInputLevel())
	fmt.Printf("%d %d %d\n", btp.CoeffsToSlotsParameters.LevelStart, btp.Mod1ParametersLiteral.LevelStart, btp.SlotsToCoeffsParameters.LevelStart)
	fmt.Printf("%d\n", params.MaxLevel())

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
		LogQ:            []int{60, 45, 45, 45, 45, 45, 42, 42, 42, 42, 42},
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

	fmt.Printf("%d  %d  %d\n", btp.Depth(), btp.OutputLevel(), btp.MinimumInputLevel())
	fmt.Printf("%d %d %d\n", btp.CoeffsToSlotsParameters.LevelStart, btp.Mod1ParametersLiteral.LevelStart, btp.SlotsToCoeffsParameters.LevelStart)

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
func Test_bootparams4(t *testing.T) {
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            16,
		LogQ:            []int{60, 45, 45, 45, 45},
		LogP:            []int{61, 61},
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
	//params = btpParams.ResidualParameters
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
	btpevk, sk2, _ := btpParams.GenEvaluationKeys(sk)

	btp, _ := bootstrapping.NewEvaluator(btpParams, btpevk)

	_, _, _, _, _ = encryptor, decryptor, encoder, evaluator, btp

	fmt.Println("end boot params")

	fmt.Printf("%d", params.MaxLevel())
	fmt.Printf("%d  %d  %d\n", btp.Depth(), btp.OutputLevel(), btp.MinimumInputLevel())
	fmt.Printf("%d %d %d\n", btp.CoeffsToSlotsParameters.LevelStart, btp.Mod1ParametersLiteral.LevelStart, btp.SlotsToCoeffsParameters.LevelStart)

	encoder2 := hefloat.NewEncoder(*btp.GetParameters())
	encryptor2 := rlwe.NewEncryptor(btp.GetParameters(), sk2)
	decryptor2 := rlwe.NewDecryptor(btp.GetParameters(), sk2)
	_, _ = encoder2, encryptor2

	n := 1 << params.LogMaxSlots()
	value := make([]complex128, n)
	for i := range value {
		if i < 100 {
			value[i] = complex(float64(i), 0)
		} else {
			value[i] = complex(0, 0)
		}
	}
	plaintext := hefloat.NewPlaintext(*btp.GetParameters(), btp.GetParameters().MaxLevel())
	encoder2.Encode(value, plaintext)
	cipher, _ := encryptor2.EncryptNew(plaintext)

	DecryptPrint(*btp.GetParameters(), cipher, *decryptor2, *encoder2)

	res, _ := btp.Bootstrap(cipher)

	DecryptPrint(params, res, *decryptor, *encoder)
}

func Test_bootparams5(t *testing.T) {
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            16,
		LogQ:            []int{51, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46},
		LogP:            []int{51, 51, 51},
		Xs:              ring.Ternary{H: 192},
		LogDefaultScale: 46,
	}
	bootstrappingParams := bootstrapping.ParametersLiteral{
		SlotsToCoeffsFactorizationDepthAndLogScales: [][]int{{51}, {51}, {51}},
		CoeffsToSlotsFactorizationDepthAndLogScales: [][]int{{51}, {51}, {51}},
		LogMessageRatio: utils.Pointy(5),
		DoubleAngle:     utils.Pointy(2),
		Mod1Degree:      utils.Pointy(63),
		K:               utils.Pointy(25),
		EvalModLogScale: utils.Pointy(51),
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
	btpevk, sk2, _ := btpParams.GenEvaluationKeys(sk)
	btp, _ := bootstrapping.NewEvaluator(btpParams, btpevk)

	_, _, _, _, _ = encryptor, decryptor, encoder, evaluator, btp

	fmt.Println("end boot params")

	fmt.Printf("%d", params.MaxLevel())
	fmt.Printf("%d  %d  %d\n", btp.Depth(), btp.OutputLevel(), btp.MinimumInputLevel())
	fmt.Printf("%d %d %d\n", btp.CoeffsToSlotsParameters.LevelStart, btp.Mod1ParametersLiteral.LevelStart, btp.SlotsToCoeffsParameters.LevelStart)
	fmt.Printf("%d %d\n\n", btp.CoeffsToSlotsParameters.LogSlots, btp.SlotsToCoeffsParameters.LogSlots)

	fmt.Printf("%d %d\n\n", btp.BootstrappingParameters.LogMaxSlots(), btp.ResidualParameters.LogMaxSlots())

	encoder2 := hefloat.NewEncoder(*btp.GetParameters())
	encryptor2 := rlwe.NewEncryptor(btp.GetParameters(), sk2)
	decryptor2 := rlwe.NewDecryptor(btp.GetParameters(), sk2)
	_, _ = encoder2, encryptor2

	n := 1 << params.LogMaxSlots()
	n2 := 1 << (params.LogMaxSlots() - 1)
	value := make([]complex128, n)
	for i := range value {
		if i < n2 {
			value[i] = sampling.RandComplex128(-1, 1)
		} else {
			value[i] = value[i%n2]
		}
	}
	starttime := time.Now()
	plaintext1 := hefloat.NewPlaintext(*btp.GetParameters(), 0)
	encoder2.Encode(value, plaintext1)
	cipher, _ := encryptor2.EncryptNew(plaintext1)
	elapse := time.Since(starttime)

	fmt.Printf("\n\n%s\n", elapse)
	DecryptPrint(*btp.GetParameters(), cipher, *decryptor2, *encoder2)
	res, _ := btp.Bootstrap(cipher)
	DecryptPrint(params, res, *decryptor, *encoder)

	meta := hefloat.NewPlaintext(params, 0).MetaData

	starttime = time.Now()
	plaintext2 := hefloat.NewPlaintext(*btp.GetParameters(), 0)
	plaintext2.MetaData = meta
	encoder.Encode(value, plaintext2)
	cipher2, _ := encryptor.EncryptNew(plaintext2)
	elapse = time.Since(starttime)

	fmt.Printf("\n\n%s\n", elapse)
	DecryptPrint(*btp.GetParameters(), cipher2, *decryptor, *encoder)
	res2, _ := btp.Bootstrap(cipher2)
	DecryptPrint(params, res2, *decryptor, *encoder)
}

func Test_bootparams6(t *testing.T) {
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

func Test_TwoSK(t *testing.T) {
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            16,
		LogQ:            []int{60, 45, 45, 45, 45, 45, 45},
		LogP:            []int{61, 61},
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
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)
	evaluator := hefloat.NewEvaluator(params, evk)
	_ = evaluator

	fmt.Println("make boot params")
	btpevk, sk2, _ := btpParams.GenEvaluationKeys(sk)
	btp, _ := bootstrapping.NewEvaluator(btpParams, btpevk)

	fmt.Println("end boot params")

	params2 := *btp.GetParameters()
	kgne2 := rlwe.NewKeyGenerator(params2)
	pk2 := kgne2.GenPublicKeyNew(sk2)
	encryptor2 := rlwe.NewEncryptor(params2, pk2)
	decryptor2 := rlwe.NewDecryptor(params2, sk2)
	encoder2 := hefloat.NewEncoder(params2)
	evaluator2 := btp.Evaluator
	_ = evaluator2

	n := 1 << params.LogMaxSlots()
	value := make([]complex128, n)
	for i := range value {
		value[i] = sampling.RandComplex128(-1, 1)
	}
	plain := hefloat.NewPlaintext(params, 3)
	plain2 := hefloat.NewPlaintext(params2, params2.MaxLevel())
	encoder.Encode(value, plain)
	encoder2.Encode(value, plain2)

	cipher1, _ := encryptor.EncryptNew(plain)
	cipher2high, _ := encryptor2.EncryptNew(plain2)
	cipher2 := evaluator2.DropLevelNew(cipher2high, params2.MaxLevel()-3)

	DecryptPrint(params, cipher1, *decryptor, *encoder)
	DecryptPrint(params2, cipher2, *decryptor2, *encoder2)

	t.Run("c1 decrypt by dec2", func(t *testing.T) {
		DecryptPrint(params2, cipher1, *decryptor2, *encoder2)
	})
	t.Run("c2 decrypt by dec1", func(t *testing.T) {
		DecryptPrint(params, cipher2, *decryptor, *encoder)
	})
	t.Run("c2 decrypt by dec1 in high level", func(t *testing.T) {
		defer func() {
			if r := recover(); r != nil {
				t.Error("Recovered from panic:", r)
			}
		}()
		DecryptPrint(params, cipher2high, *decryptor, *encoder)
	})

	t.Run("c2 decrypt by dec1 in high level(param2)", func(t *testing.T) {
		defer func() {
			if r := recover(); r != nil {
				t.Error("Recovered from panic:", r)
			}
		}()
		DecryptPrint(params2, cipher2high, *decryptor, *encoder)
	})
	t.Run("c2 decrypt by dec1 in high level(param2, dec2)", func(t *testing.T) {
		defer func() {
			if r := recover(); r != nil {
				t.Error("Recovered from panic:", r)
			}
		}()
		DecryptPrint(params2, cipher2high, *decryptor2, *encoder)
	})
	t.Run("c2 decrypt by dec1 in high level(param2, dec2, enc2)", func(t *testing.T) {
		defer func() {
			if r := recover(); r != nil {
				t.Error("Recovered from panic:", r)
			}
		}()
		DecryptPrint(params2, cipher2high, *decryptor2, *encoder2)
	})

	t.Run("c2 decrypt by dec1 in high level(param2, dec2, enc2)", func(t *testing.T) {
		defer func() {
			if r := recover(); r != nil {
				t.Error("Recovered from panic:", r)
			}
		}()
		DecryptPrint(params2, cipher2high, *decryptor2, *encoder2)
	})
}

func Test_bootparams7(t *testing.T) {
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	SchemeParams := cnn.CNN_Cifar10_Parameters.SchemeParams
	bootstrappingParams := cnn.CNN_Cifar10_Parameters.BootstrappingParams
	bootstrappingParams.LogN = utils.Pointy(17)
	SchemeParams.LogN = 17

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

	fmt.Printf("%d  %d  %d\n", btp.Depth(), btp.OutputLevel(), btp.MinimumInputLevel())
	fmt.Printf("%d %d %d\n", btp.CoeffsToSlotsParameters.LevelStart, btp.Mod1ParametersLiteral.LevelStart, btp.SlotsToCoeffsParameters.LevelStart)
	fmt.Printf("%d\n", params.MaxLevel())

	n := 1 << params.LogMaxSlots()
	value := make([]complex128, n)
	for i := range value {
		value[i] = sampling.RandComplex128(-1, 1)
	}
	plaintext := hefloat.NewPlaintext(params, params.MaxLevel())
	encoder.Encode(value, plaintext)
	cipher, _ := encryptor.EncryptNew(plaintext)
	evaluator.DropLevel(cipher, params.MaxLevel())
	DecryptPrint(params, cipher, *decryptor, *encoder)

	res, _ := btp.Bootstrap(cipher)

	DecryptPrint(params, res, *decryptor, *encoder)
}
