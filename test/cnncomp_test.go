package test

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/lifejade/cnn_ckksv5/cnn"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"
	"github.com/tuneinsight/lattigo/v5/utils/sampling"
)

func Test_Conv2(t *testing.T) {
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	os.Chdir("../")

	ckksParams := cnn.CNN_Cifar10_Parameters

	//parameter init
	params, err := hefloat.NewParametersFromLiteral(ckksParams.SchemeParams)
	if err != nil {
		panic(err)
	}
	btpParams, err := bootstrapping.NewParametersFromLiteral(params, ckksParams.BootstrappingParams)
	if err != nil {
		panic(err)
	}
	keygen := rlwe.NewKeyGenerator(params)
	sk := keygen.GenSecretKeyNew()

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey
	btpevk, _, _ := btpParams.GenEvaluationKeys(sk)
	btp, _ := bootstrapping.NewEvaluator(btpParams, btpevk)
	_ = btp
	kgen := rlwe.NewKeyGenerator(params)
	pk = kgen.GenPublicKeyNew(sk)
	rlk = kgen.GenRelinearizationKeyNew(sk)
	// generate keys - Rotating key
	convRot := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
		34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
		56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 84, 124, 128, 132, 256, 512, 959, 960, 990, 991, 1008,
		1023, 1024, 1036, 1064, 1092, 1952, 1982, 1983, 2016, 2044, 2047, 2048, 2072, 2078, 2100, 3007, 3024, 3040, 3052, 3070, 3071, 3072, 3080, 3108, 4031,
		4032, 4062, 4063, 4095, 4096, 5023, 5024, 5054, 5055, 5087, 5118, 5119, 5120, 6047, 6078, 6079, 6111, 6112, 6142, 6143, 6144, 7071, 7102, 7103, 7135,
		7166, 7167, 7168, 8095, 8126, 8127, 8159, 8190, 8191, 8192, 9149, 9183, 9184, 9213, 9215, 9216, 10173, 10207, 10208, 10237, 10239, 10240, 11197, 11231,
		11232, 11261, 11263, 11264, 12221, 12255, 12256, 12285, 12287, 12288, 13214, 13216, 13246, 13278, 13279, 13280, 13310, 13311, 13312, 14238, 14240,
		14270, 14302, 14303, 14304, 14334, 14335, 15262, 15264, 15294, 15326, 15327, 15328, 15358, 15359, 15360, 16286, 16288, 16318, 16350, 16351, 16352,
		16382, 16383, 16384, 17311, 17375, 18335, 18399, 18432, 19359, 19423, 20383, 20447, 20480, 21405, 21406, 21437, 21469, 21470, 21471, 21501, 21504,
		22429, 22430, 22461, 22493, 22494, 22495, 22525, 22528, 23453, 23454, 23485, 23517, 23518, 23519, 23549, 24477, 24478, 24509, 24541, 24542, 24543,
		24573, 24576, 25501, 25565, 25568, 25600, 26525, 26589, 26592, 26624, 27549, 27613, 27616, 27648, 28573, 28637, 28640, 28672, 29600, 29632, 29664,
		29696, 30624, 30656, 30688, 30720, 31648, 31680, 31712, 31743, 31744, 31774, 32636, 32640, 32644, 32672, 32702, 32704, 32706, 32735,
		32736, 32737, 32759, 32760, 32761, 32762, 32763, 32764, 32765, 32766, 32767}
	galEls := make([]uint64, len(convRot))
	for i, x := range convRot {
		galEls[i] = params.GaloisElement(x)
	}
	galEls = append(galEls, params.GaloisElementForComplexConjugation())

	rtk = make([]*rlwe.GaloisKey, len(galEls))
	var wg sync.WaitGroup
	wg.Add(len(galEls))
	for i := range galEls {
		i := i
		go func() {
			defer wg.Done()
			kgen_ := rlwe.NewKeyGenerator(params)
			rtk[i] = kgen_.GenGaloisKeyNew(galEls[i], sk)
		}()
	}
	wg.Wait()
	evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)
	//generate -er
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)
	evaluator := hefloat.NewEvaluator(params, evk)
	fmt.Println("generate Evaluator end")
	context := cnn.NewContext(encoder, encryptor, decryptor, sk, pk, btp, btp, btp, rtk, rlk, evaluator, &params)

	file, err := os.Open("testFiles/test_values.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	in := bufio.NewScanner(file)
	cnn.SkipLines(in, 32*32*3*0)
	var image []float64
	n := 1 << 15
	image = cnn.ReadLines(in, image, 32*32*3)
	for len(image) < n {
		image = append(image, 0.0)
	}
	for i := n / 8; i < n; i++ {
		image[i] = image[i%(n/8)]
	}
	for i := 0; i < n; i++ {
		image[i] /= 40
	}

	_, _, convWgt, _, _, bnVar, bnWgt, endNum := cnn.ImportParametersCifar10(20)

	largeBlockID, blockID := 0, 0
	co, st := 16, 1
	stage := 2*((endNum+1)*largeBlockID+blockID) + 1
	fmt.Printf("\n//////////////////////\nco = %d, stage = %d\n", co, stage)
	conv2(largeBlockID, image, context, evaluator, convWgt, stage, co, st, bnVar, bnWgt, params, decryptor, encoder)

	largeBlockID, blockID = 1, 0
	co, st = 32, 2
	stage = 2*((endNum+1)*largeBlockID+blockID) + 1
	fmt.Printf("\n//////////////////////\nco = %d, stage = %d\n", co, stage)
	conv2(largeBlockID, image, context, evaluator, convWgt, stage, co, st, bnVar, bnWgt, params, decryptor, encoder)

	largeBlockID, blockID = 2, 0
	co, st = 64, 2
	stage = 2*((endNum+1)*largeBlockID+blockID) + 1
	fmt.Printf("\n//////////////////////\nco = %d, stage = %d\n", co, stage)
	conv2(largeBlockID, image, context, evaluator, convWgt, stage, co, st, bnVar, bnWgt, params, decryptor, encoder)

	largeBlockID, blockID = 2, 1
	co, st = 64, 1
	stage = 2*((endNum+1)*largeBlockID+blockID) + 1
	fmt.Printf("\n//////////////////////\nco = %d, stage = %d\n", co, stage)
	conv2(largeBlockID, image, context, evaluator, convWgt, stage, co, st, bnVar, bnWgt, params, decryptor, encoder)

}
func Test_Conv(t *testing.T) {
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	os.Chdir("../")

	ckksParams := cnn.CNN_Cifar10_Parameters

	//parameter init
	params, err := hefloat.NewParametersFromLiteral(ckksParams.SchemeParams)
	if err != nil {
		panic(err)
	}
	btpParams, err := bootstrapping.NewParametersFromLiteral(params, ckksParams.BootstrappingParams)
	if err != nil {
		panic(err)
	}
	keygen := rlwe.NewKeyGenerator(params)
	sk := keygen.GenSecretKeyNew()

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey
	btpevk, _, _ := btpParams.GenEvaluationKeys(sk)
	btp, _ := bootstrapping.NewEvaluator(btpParams, btpevk)
	_ = btp
	kgen := rlwe.NewKeyGenerator(params)
	pk = kgen.GenPublicKeyNew(sk)
	rlk = kgen.GenRelinearizationKeyNew(sk)
	// generate keys - Rotating key
	convRot := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
		34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
		56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 84, 124, 128, 132, 256, 512, 959, 960, 990, 991, 1008,
		1023, 1024, 1036, 1064, 1092, 1952, 1982, 1983, 2016, 2044, 2047, 2048, 2072, 2078, 2100, 3007, 3024, 3040, 3052, 3070, 3071, 3072, 3080, 3108, 4031,
		4032, 4062, 4063, 4095, 4096, 5023, 5024, 5054, 5055, 5087, 5118, 5119, 5120, 6047, 6078, 6079, 6111, 6112, 6142, 6143, 6144, 7071, 7102, 7103, 7135,
		7166, 7167, 7168, 8095, 8126, 8127, 8159, 8190, 8191, 8192, 9149, 9183, 9184, 9213, 9215, 9216, 10173, 10207, 10208, 10237, 10239, 10240, 11197, 11231,
		11232, 11261, 11263, 11264, 12221, 12255, 12256, 12285, 12287, 12288, 13214, 13216, 13246, 13278, 13279, 13280, 13310, 13311, 13312, 14238, 14240,
		14270, 14302, 14303, 14304, 14334, 14335, 15262, 15264, 15294, 15326, 15327, 15328, 15358, 15359, 15360, 16286, 16288, 16318, 16350, 16351, 16352,
		16382, 16383, 16384, 17311, 17375, 18335, 18399, 18432, 19359, 19423, 20383, 20447, 20480, 21405, 21406, 21437, 21469, 21470, 21471, 21501, 21504,
		22429, 22430, 22461, 22493, 22494, 22495, 22525, 22528, 23453, 23454, 23485, 23517, 23518, 23519, 23549, 24477, 24478, 24509, 24541, 24542, 24543,
		24573, 24576, 25501, 25565, 25568, 25600, 26525, 26589, 26592, 26624, 27549, 27613, 27616, 27648, 28573, 28637, 28640, 28672, 29600, 29632, 29664,
		29696, 30624, 30656, 30688, 30720, 31648, 31680, 31712, 31743, 31744, 31774, 32636, 32640, 32644, 32672, 32702, 32704, 32706, 32735,
		32736, 32737, 32759, 32760, 32761, 32762, 32763, 32764, 32765, 32766, 32767}
	galEls := make([]uint64, len(convRot))
	for i, x := range convRot {
		galEls[i] = params.GaloisElement(x)
	}
	galEls = append(galEls, params.GaloisElementForComplexConjugation())

	rtk = make([]*rlwe.GaloisKey, len(galEls))
	var wg sync.WaitGroup
	wg.Add(len(galEls))
	for i := range galEls {
		i := i
		go func() {
			defer wg.Done()
			kgen_ := rlwe.NewKeyGenerator(params)
			rtk[i] = kgen_.GenGaloisKeyNew(galEls[i], sk)
		}()
	}
	wg.Wait()
	evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)
	//generate -er
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)
	evaluator := hefloat.NewEvaluator(params, evk)
	fmt.Println("generate Evaluator end")
	context := cnn.NewContext(encoder, encryptor, decryptor, sk, pk, btp, btp, btp, rtk, rlk, evaluator, &params)

	file, err := os.Open("testFiles/test_values.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	in := bufio.NewScanner(file)
	cnn.SkipLines(in, 32*32*3*0)
	var image []float64
	n := 1 << 15
	image = cnn.ReadLines(in, image, 32*32*3)
	for len(image) < n {
		image = append(image, 0.0)
	}
	for i := n / 8; i < n; i++ {
		image[i] = image[i%(n/8)]
	}
	for i := 0; i < n; i++ {
		image[i] /= 40
	}

	_, _, convWgt, _, _, bnVar, bnWgt, endNum := cnn.ImportParametersCifar10(20)

	largeBlockID, blockID := 0, 0
	co, st := 16, 1
	stage := 2*((endNum+1)*largeBlockID+blockID) + 1
	fmt.Printf("\n//////////////////////\nco = %d, stage = %d\n", co, stage)
	conv(largeBlockID, image, context, evaluator, convWgt, stage, co, st, bnVar, bnWgt, params, decryptor, encoder)

	largeBlockID, blockID = 1, 0
	co, st = 32, 2
	stage = 2*((endNum+1)*largeBlockID+blockID) + 1
	fmt.Printf("\n//////////////////////\nco = %d, stage = %d\n", co, stage)
	conv(largeBlockID, image, context, evaluator, convWgt, stage, co, st, bnVar, bnWgt, params, decryptor, encoder)

	largeBlockID, blockID = 2, 0
	co, st = 64, 2
	stage = 2*((endNum+1)*largeBlockID+blockID) + 1
	fmt.Printf("\n//////////////////////\nco = %d, stage = %d\n", co, stage)
	conv(largeBlockID, image, context, evaluator, convWgt, stage, co, st, bnVar, bnWgt, params, decryptor, encoder)

	largeBlockID, blockID = 2, 1
	co, st = 64, 1
	stage = 2*((endNum+1)*largeBlockID+blockID) + 1
	fmt.Printf("\n//////////////////////\nco = %d, stage = %d\n", co, stage)
	conv(largeBlockID, image, context, evaluator, convWgt, stage, co, st, bnVar, bnWgt, params, decryptor, encoder)

}

func conv(largeID int, image []float64, context cnn.Context, evaluator *hefloat.Evaluator, convWgt [][]float64, stage int, co int, st int, bnVar [][]float64, bnWgt [][]float64, params hefloat.Parameters, decryptor *rlwe.Decryptor, encoder *hefloat.Encoder) {
	var k, hw, c, t, p int
	if largeID == 2 {
		k, hw, c, t, p = 2, 16, 32, 8, 4
		if stage == 15 {
			c = 64
		}
	} else if largeID == 1 {
		k, hw, c, t, p = 1, 32, 16, 16, 2
	} else if largeID == 0 {
		k, hw, c, t, p = 1, 32, 16, 16, 2
	}
	cnn2 := cnn.NewTensorCipherFormData(k, hw, hw, c, t, p, 15, 46, image, context)
	cipher := cnn.Get_Cipher(cnn2)
	evaluator.DropLevel(cipher, cipher.Level()-2)
	cnn.DecryptPrint(params, cipher, *decryptor, *encoder)
	startTime := time.Now()
	cnn2 = cnn.CompactGappedConvolution(cnn2, co, st, 3, 3, convWgt[stage], bnVar[stage], bnWgt[stage], 0.00001, &context)
	elapse := time.Since(startTime)
	fmt.Printf("low level : %s\n", elapse)
}

func conv2(largeID int, image []float64, context cnn.Context, evaluator *hefloat.Evaluator, convWgt [][]float64, stage int, co int, st int, bnVar [][]float64, bnWgt [][]float64, params hefloat.Parameters, decryptor *rlwe.Decryptor, encoder *hefloat.Encoder) {
	var k, hw, c, t, p int
	if largeID == 2 {
		k, hw, c, t, p = 2, 16, 32, 8, 4
		if stage == 15 {
			c = 64
		}
	} else if largeID == 1 {
		k, hw, c, t, p = 1, 32, 16, 16, 2
	} else if largeID == 0 {
		k, hw, c, t, p = 1, 32, 16, 16, 2
	}
	cnn2 := cnn.NewTensorCipherFormData(k, hw, hw, c, t, p, 15, 46, image, context)
	cipher := cnn.Get_Cipher(cnn2)
	evaluator.DropLevel(cipher, cipher.Level()-2)
	cnn.DecryptPrint(params, cipher, *decryptor, *encoder)
	startTime := time.Now()
	cnn2 = cnn.CompactGappedConvolution2(cnn2, co, st, 3, 3, convWgt[stage], bnVar[stage], bnWgt[stage], 0.00001, &context)
	elapse := time.Since(startTime)
	fmt.Printf("low level : %s\n", elapse)
}

func Test_multime(t *testing.T) {
	params, err := hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{
		LogN:            16,
		LogQ:            []int{51, 46, 46, 46, 46, 46, 46, 46, 46},
		LogP:            []int{51, 51, 51, 51, 51},
		LogDefaultScale: 46,
	})
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

	n := params.MaxSlots()
	values := make([]float64, n)
	for i := range values {
		values[i] = complex(-0.8, 0)
	}
	plain := hefloat.NewPlaintext(params, 2)
	encoder.Encode(values, plain)
	cipher, _ := encryptor.EncryptNew(plain)
	add := cipher.CopyNew()

	startTime := time.Now()
	for i := 0; i < 9; i++ {
		temp, _ := evaluator.MulNew(cipher, values)
		evaluator.Add(add, temp, add)
	}
	elapse := time.Since(startTime)
	fmt.Printf("%s", elapse)
	_ = decryptor
}

func Test_multime2(t *testing.T) {
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	os.Chdir("../")

	ckksParams := cnn.CNN_Cifar10_Parameters

	//parameter init
	params, err := hefloat.NewParametersFromLiteral(ckksParams.SchemeParams)
	if err != nil {
		panic(err)
	}
	btpParams, err := bootstrapping.NewParametersFromLiteral(params, ckksParams.BootstrappingParams)
	if err != nil {
		panic(err)
	}
	keygen_ := rlwe.NewKeyGenerator(params)
	sk_ := keygen_.GenSecretKeyNew()

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey
	btpevk, sk, _ := btpParams.GenEvaluationKeys(sk_)
	btp, _ := bootstrapping.NewEvaluator(btpParams, btpevk)
	params = *btp.GetParameters()

	kgen := rlwe.NewKeyGenerator(params)
	pk = kgen.GenPublicKeyNew(sk)
	rlk = kgen.GenRelinearizationKeyNew(sk)
	// generate keys - Rotating key
	convRot := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
		34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
		56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 84, 124, 128, 132, 256, 512, 959, 960, 990, 991, 1008,
		1023, 1024, 1036, 1064, 1092, 1952, 1982, 1983, 2016, 2044, 2047, 2048, 2072, 2078, 2100, 3007, 3024, 3040, 3052, 3070, 3071, 3072, 3080, 3108, 4031,
		4032, 4062, 4063, 4095, 4096, 5023, 5024, 5054, 5055, 5087, 5118, 5119, 5120, 6047, 6078, 6079, 6111, 6112, 6142, 6143, 6144, 7071, 7102, 7103, 7135,
		7166, 7167, 7168, 8095, 8126, 8127, 8159, 8190, 8191, 8192, 9149, 9183, 9184, 9213, 9215, 9216, 10173, 10207, 10208, 10237, 10239, 10240, 11197, 11231,
		11232, 11261, 11263, 11264, 12221, 12255, 12256, 12285, 12287, 12288, 13214, 13216, 13246, 13278, 13279, 13280, 13310, 13311, 13312, 14238, 14240,
		14270, 14302, 14303, 14304, 14334, 14335, 15262, 15264, 15294, 15326, 15327, 15328, 15358, 15359, 15360, 16286, 16288, 16318, 16350, 16351, 16352,
		16382, 16383, 16384, 17311, 17375, 18335, 18399, 18432, 19359, 19423, 20383, 20447, 20480, 21405, 21406, 21437, 21469, 21470, 21471, 21501, 21504,
		22429, 22430, 22461, 22493, 22494, 22495, 22525, 22528, 23453, 23454, 23485, 23517, 23518, 23519, 23549, 24477, 24478, 24509, 24541, 24542, 24543,
		24573, 24576, 25501, 25565, 25568, 25600, 26525, 26589, 26592, 26624, 27549, 27613, 27616, 27648, 28573, 28637, 28640, 28672, 29600, 29632, 29664,
		29696, 30624, 30656, 30688, 30720, 31648, 31680, 31712, 31743, 31744, 31774, 32636, 32640, 32644, 32672, 32702, 32704, 32706, 32735,
		32736, 32737, 32759, 32760, 32761, 32762, 32763, 32764, 32765, 32766, 32767}
	galEls := make([]uint64, len(convRot))
	for i, x := range convRot {
		galEls[i] = params.GaloisElement(x)
	}
	galEls = append(galEls, params.GaloisElementForComplexConjugation())

	rtk = make([]*rlwe.GaloisKey, len(galEls))
	var wg sync.WaitGroup
	wg.Add(len(galEls))
	for i := range galEls {
		i := i
		go func() {
			defer wg.Done()
			kgen__ := rlwe.NewKeyGenerator(params)
			rtk[i] = kgen__.GenGaloisKeyNew(galEls[i], sk)
		}()
	}
	wg.Wait()
	evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)
	//generate -er
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)
	evaluator := hefloat.NewEvaluator(params, evk)
	fmt.Println("generate Evaluator end")

	n := params.MaxSlots()
	values := make([]complex128, n)
	for i := range values {
		values[i] = sampling.RandComplex128(-1.0, 1.0)
	}
	fmt.Printf("max level : %d\n", params.MaxLevel())
	plain := hefloat.NewPlaintext(params, 2)
	encoder.Encode(values, plain)
	cipher, _ := encryptor.EncryptNew(plain)

	plainconst := hefloat.NewPlaintext(params, cipher.Level())
	plainconst.Scale = rlwe.NewScale(cipher.LevelQ())

	add := cipher.CopyNew()
	startTime := time.Now()
	for i := 0; i < 9; i++ {
		encoder.Embed(values, plainconst.MetaData, plainconst.Value)
		temp, _ := evaluator.MulNew(cipher, plainconst)
		evaluator.Add(add, temp, add)
	}
	elapse := time.Since(startTime)
	fmt.Printf("%s\n", elapse)
	_ = decryptor

	const_value := make([]complex128, n)
	for i := range const_value {
		const_value[i] = complex(-0.8, 0)
	}
	add = cipher.CopyNew()
	startTime = time.Now()
	for i := 0; i < 9; i++ {
		encoder.Embed(const_value, plainconst.MetaData, plainconst.Value)
		temp, _ := evaluator.MulNew(cipher, plainconst)
		evaluator.Add(add, temp, add)
	}
	elapse = time.Since(startTime)
	fmt.Printf("%s\n", elapse)
	_ = decryptor
}

func Test_multimeThread(t *testing.T) {
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	os.Chdir("../")

	ckksParams := cnn.CNN_Cifar10_Parameters

	//parameter init
	params, err := hefloat.NewParametersFromLiteral(ckksParams.SchemeParams)
	if err != nil {
		panic(err)
	}
	btpParams, err := bootstrapping.NewParametersFromLiteral(params, ckksParams.BootstrappingParams)
	if err != nil {
		panic(err)
	}
	keygen_ := rlwe.NewKeyGenerator(params)
	sk_ := keygen_.GenSecretKeyNew()

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey
	btpevk, sk, _ := btpParams.GenEvaluationKeys(sk_)
	btp, _ := bootstrapping.NewEvaluator(btpParams, btpevk)
	params = *btp.GetParameters()

	kgen := rlwe.NewKeyGenerator(params)
	pk = kgen.GenPublicKeyNew(sk)
	rlk = kgen.GenRelinearizationKeyNew(sk)
	// generate keys - Rotating key
	convRot := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
		34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
		56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 84, 124, 128, 132, 256, 512, 959, 960, 990, 991, 1008,
		1023, 1024, 1036, 1064, 1092, 1952, 1982, 1983, 2016, 2044, 2047, 2048, 2072, 2078, 2100, 3007, 3024, 3040, 3052, 3070, 3071, 3072, 3080, 3108, 4031,
		4032, 4062, 4063, 4095, 4096, 5023, 5024, 5054, 5055, 5087, 5118, 5119, 5120, 6047, 6078, 6079, 6111, 6112, 6142, 6143, 6144, 7071, 7102, 7103, 7135,
		7166, 7167, 7168, 8095, 8126, 8127, 8159, 8190, 8191, 8192, 9149, 9183, 9184, 9213, 9215, 9216, 10173, 10207, 10208, 10237, 10239, 10240, 11197, 11231,
		11232, 11261, 11263, 11264, 12221, 12255, 12256, 12285, 12287, 12288, 13214, 13216, 13246, 13278, 13279, 13280, 13310, 13311, 13312, 14238, 14240,
		14270, 14302, 14303, 14304, 14334, 14335, 15262, 15264, 15294, 15326, 15327, 15328, 15358, 15359, 15360, 16286, 16288, 16318, 16350, 16351, 16352,
		16382, 16383, 16384, 17311, 17375, 18335, 18399, 18432, 19359, 19423, 20383, 20447, 20480, 21405, 21406, 21437, 21469, 21470, 21471, 21501, 21504,
		22429, 22430, 22461, 22493, 22494, 22495, 22525, 22528, 23453, 23454, 23485, 23517, 23518, 23519, 23549, 24477, 24478, 24509, 24541, 24542, 24543,
		24573, 24576, 25501, 25565, 25568, 25600, 26525, 26589, 26592, 26624, 27549, 27613, 27616, 27648, 28573, 28637, 28640, 28672, 29600, 29632, 29664,
		29696, 30624, 30656, 30688, 30720, 31648, 31680, 31712, 31743, 31744, 31774, 32636, 32640, 32644, 32672, 32702, 32704, 32706, 32735,
		32736, 32737, 32759, 32760, 32761, 32762, 32763, 32764, 32765, 32766, 32767}
	galEls := make([]uint64, len(convRot))
	for i, x := range convRot {
		galEls[i] = params.GaloisElement(x)
	}
	galEls = append(galEls, params.GaloisElementForComplexConjugation())

	rtk = make([]*rlwe.GaloisKey, len(galEls))
	var wg sync.WaitGroup
	wg.Add(len(galEls))
	for i := range galEls {
		i := i
		go func() {
			defer wg.Done()
			kgen__ := rlwe.NewKeyGenerator(params)
			rtk[i] = kgen__.GenGaloisKeyNew(galEls[i], sk)
		}()
	}
	wg.Wait()
	evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)
	//generate -er
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)
	evaluator := hefloat.NewEvaluator(params, evk)
	fmt.Println("generate Evaluator end")

	n := params.MaxSlots()
	values := make([]complex128, n)
	for i := range values {
		values[i] = sampling.RandComplex128(-1.0, 1.0)
	}
	fmt.Printf("max level : %d\n", params.MaxLevel())
	plain := hefloat.NewPlaintext(params, 2)
	encoder.Encode(values, plain)
	cipher, _ := encryptor.EncryptNew(plain)

	add := cipher.CopyNew()
	startTime := time.Now()
	var wg2 sync.WaitGroup
	wg2.Add(9)
	for i := 0; i < 9; i++ {

		go func() {
			defer wg2.Done()
			temp, _ := evaluator.MulNew(cipher, values)
			evaluator.Add(add, temp, add)
		}()

	}
	wg2.Wait()
	elapse := time.Since(startTime)
	fmt.Printf("%s\n", elapse)
	_ = decryptor
}
