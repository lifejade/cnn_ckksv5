package cnn

import (
	"fmt"
	"runtime"

	"bufio"
	"log"
	"os"
	"strconv"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"
	"github.com/tuneinsight/lattigo/v5/utils"
)

func ResNetCifar10(layerNum int, imageID int) {
	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

	//check layernumber
	if !(layerNum == 20 || layerNum == 32 || layerNum == 44 || layerNum == 56 || layerNum == 110) {
		fmt.Println("layer_num is not correct")
		os.Exit(1)
	}
	// load inference image
	var file *os.File
	file, err := os.Open("/home/user/testFile/test_values.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	in := bufio.NewScanner(file)
	SkipLines(in, 32*32*3*imageID)
	var image []float64
	image = ReadLines(in, image, 32*32*3)

	//result file init
	var result_file *os.File
	result_file, _ = os.Create("../result/resnet" + strconv.Itoa(layerNum) + "_cifar10_label" + strconv.Itoa(imageID) + ".txt")
	outShare := bufio.NewWriter(result_file)
	_ = outShare
	fmt.Println("Resnet-CKKS result name: ", result_file.Name())

	//ckks parameter init
	ckksParams := CNN_Cifar10_Parameters
	params, err := hefloat.NewParametersFromLiteral(ckksParams.SchemeParams)
	if err != nil {
		panic(err)
	}
	fmt.Println("ckks parameter init end")

	//boot parameter init
	bootParams14, bootParams13, bootParams12 := ckksParams.BootstrappingParams, ckksParams.BootstrappingParams, ckksParams.BootstrappingParams
	//Todo : not RS-Bootstrapping yet
	bootParams14.LogSlots, bootParams13.LogSlots, bootParams12.LogSlots = utils.Pointy(15), utils.Pointy(15), utils.Pointy(15)
	btpParams14, err := bootstrapping.NewParametersFromLiteral(params, bootParams14)
	if err != nil {
		panic(err)
	}
	btpParams13, err := bootstrapping.NewParametersFromLiteral(params, bootParams13)
	if err != nil {
		panic(err)
	}
	btpParams12, err := bootstrapping.NewParametersFromLiteral(params, bootParams12)
	if err != nil {
		panic(err)
	}
	fmt.Println("bootstrapping parameter init end")

	// generate keys
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	//Todo : temp rotation keys
	convRot := []int{0, 1, 2, 3, 4, 5, 6}
	galEls := make([]uint64, len(convRot))
	for i, x := range convRot {
		galEls[i] = params.GaloisElement(x)
	}
	rlk := kgen.GenRelinearizationKeyNew(sk)
	rtk := kgen.GenGaloisKeysNew(galEls, sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)
	fmt.Println("generate key end")

	//generate -or/er
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)
	evaluator := hefloat.NewEvaluator(params, evk)
	fmt.Println("generate Evaluator end")

	_, _, _, _ = encryptor, decryptor, encoder, evaluator

	//generate bootstrapper
	btpevk14, _, _ := btpParams14.GenEvaluationKeys(sk)
	btp14, _ := bootstrapping.NewEvaluator(btpParams14, btpevk14)
	btpevk13, _, _ := btpParams13.GenEvaluationKeys(sk)
	btp13, _ := bootstrapping.NewEvaluator(btpParams13, btpevk13)
	btpevk12, _, _ := btpParams12.GenEvaluationKeys(sk)
	btp12, _ := bootstrapping.NewEvaluator(btpParams12, btpevk12)
	_, _, _ = btp14, btp13, btp12

	fmt.Println("generated bootstrapper end")

	//Resnet parameter init
	co, st, fh, fw := 0, 0, 3, 3
	init_p := 8
	logp := 46
	logq := 51
	logn := 15
	n := 1 << logn
	stage := 0
	epsilon := 0.00001
	B := 40.0
	linWgt, linBias, convWgt, bnBias, bnMean, bnVar, bnWgt, endNum := ImportParametersCifar10(layerNum)
	_, _, _, _ = co, st, fh, fw
	_, _ = logp, logq
	_, _ = stage, epsilon
	_, _, _, _, _, _, _, _ = linWgt, linBias, convWgt, bnBias, bnMean, bnVar, bnWgt, endNum

	//preprocess inference image
	for len(image) < n {
		image = append(image, 0.0)
	}
	for i := n / init_p; i < n; i++ { // copy and paste
		image[i] = image[i%(n/init_p)]
	}
	for i := 0; i < n; i++ {
		image[i] /= B
	}
	context := NewContext(encoder, encryptor, decryptor, sk, pk, btp14, btp13, btp12, rtk, rlk, evaluator, &params)
	cnn := NewTensorCipherFormData(1, 32, 32, 3, 3, init_p, logn, image, context)
	_ = cnn
	fmt.Println("preprocess & encrypt inference image end")

}

func ImportParametersCifar10(layerNum int) (linwgt []float64, linbias []float64, convwgt [][]float64, bnbias [][]float64, bnmean [][]float64, bnvar [][]float64, bnwgt [][]float64, endNum int) {

	var dir string

	// directory name
	if layerNum != 20 && layerNum != 32 && layerNum != 44 && layerNum != 56 && layerNum != 110 {
		fmt.Println("layer number is not valid")
	}
	if layerNum == 20 {
		dir = "resnet20_new"
	} else if layerNum == 32 {
		dir = "resnet32_new"
	} else if layerNum == 44 {
		dir = "resnet44_new"
	} else if layerNum == 56 {
		dir = "resnet56_new"
	} else if layerNum == 110 {
		dir = "resnet110_new"
	} else {
		dir = ""
	}

	// endNum
	if layerNum == 20 {
		endNum = 2
	} else if layerNum == 32 {
		endNum = 4
	} else if layerNum == 44 {
		endNum = 6
	} else if layerNum == 56 {
		endNum = 8
	} else if layerNum == 110 {
		endNum = 17
	} else {
		fmt.Println("layer_num is not correct")
	}

	var num_c, num_b, num_m, num_v, num_w int
	// var num_c int

	convwgt = make([][]float64, layerNum-1)
	bnbias = make([][]float64, layerNum-1)
	bnmean = make([][]float64, layerNum-1)
	bnvar = make([][]float64, layerNum-1)
	bnwgt = make([][]float64, layerNum-1)

	fh, fw, ci, co := 3, 3, 0, 0

	// convolution parameters
	ci = 3
	co = 16
	ReadLinesIdx("../../../../../../../pretrained_parameters/"+dir+"/conv1_weight.txt", convwgt, fh*fw*ci*co, num_c)
	num_c++

	// convolution parameters
	for j := 1; j <= 3; j++ {
		for k := 0; k <= endNum; k++ {
			// co setting
			if j == 1 {
				co = 16
			} else if j == 2 {
				co = 32
			} else if j == 3 {
				co = 64
			}

			// ci setting
			if j == 1 || (j == 2 && k == 0) {
				ci = 16
			} else if (j == 2 && k != 0) || (j == 3 && k == 0) {
				ci = 32
			} else {
				ci = 64
			}
			ReadLinesIdx("../../../../../../../pretrained_parameters/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_conv1_weight.txt", convwgt, fh*fw*ci*co, num_c)
			num_c++

			// ci setting
			if j == 1 {
				ci = 16
			} else if j == 2 {
				ci = 32
			} else if j == 3 {
				ci = 64
			}
			ReadLinesIdx("../../../../../../../pretrained_parameters/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_conv2_weight.txt", convwgt, fh*fw*ci*co, num_c)
			num_c++
		}
	}

	// batch_normalization parameters
	ci = 16
	ReadLinesIdx("../../../../../../../pretrained_parameters/"+dir+"/bn1_bias.txt", bnbias, ci, num_b)
	num_b++
	ReadLinesIdx("../../../../../../../pretrained_parameters/"+dir+"/bn1_running_mean.txt", bnmean, ci, num_m)
	num_m++
	ReadLinesIdx("../../../../../../../pretrained_parameters/"+dir+"/bn1_running_var.txt", bnvar, ci, num_v)
	num_v++
	ReadLinesIdx("../../../../../../../pretrained_parameters/"+dir+"/bn1_weight.txt", bnwgt, ci, num_w)
	num_w++

	// batch_normalization parameters
	for j := 1; j <= 3; j++ {
		if j == 1 {
			ci = 16
		} else if j == 2 {
			ci = 32
		} else if j == 3 {
			ci = 64
		}

		for k := 0; k <= endNum; k++ {
			ReadLinesIdx("../../../../../../../pretrained_parameters/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_bn1_bias.txt", bnbias, ci, num_b)
			num_b++
			ReadLinesIdx("../../../../../../../pretrained_parameters/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_bn1_running_mean.txt", bnmean, ci, num_m)
			num_m++
			ReadLinesIdx("../../../../../../../pretrained_parameters/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_bn1_running_var.txt", bnvar, ci, num_v)
			num_v++
			ReadLinesIdx("../../../../../../../pretrained_parameters/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_bn1_weight.txt", bnwgt, ci, num_w)
			num_w++
			ReadLinesIdx("../../../../../../../pretrained_parameters/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_bn2_bias.txt", bnbias, ci, num_b)
			num_b++
			ReadLinesIdx("../../../../../../../pretrained_parameters/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_bn2_running_mean.txt", bnmean, ci, num_m)
			num_m++
			ReadLinesIdx("../../../../../../../pretrained_parameters/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_bn2_running_var.txt", bnvar, ci, num_v)
			num_v++
			ReadLinesIdx("../../../../../../../pretrained_parameters/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_bn2_weight.txt", bnwgt, ci, num_w)
			num_w++
		}
	}
	// FC layer
	file, err := os.Open("../../../../../../../pretrained_parameters/" + dir + "/linear_weight.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	in := bufio.NewScanner(file)
	linwgt = ReadLines(in, linwgt, 10*64)
	file, err = os.Open("../../../../../../../pretrained_parameters/" + dir + "/linear_bias.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	in = bufio.NewScanner(file)
	linbias = ReadLines(in, linbias, 10)

	return linwgt, linbias, convwgt, bnbias, bnmean, bnvar, bnwgt, endNum
}

func DecryptPrint(params hefloat.Parameters, ciphertext *rlwe.Ciphertext, decryptor rlwe.Decryptor, encoder hefloat.Encoder) {

	n := 1 << ciphertext.LogSlots()
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

func ReadLinesIdx(fileName string, storeVal [][]float64, lineNum int, idx int) {
	in, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer in.Close()

	scanner := bufio.NewScanner(in)
	for i := 0; i < lineNum; i++ {
		scanner.Scan()
		s := scanner.Text()
		p, _ := strconv.ParseFloat(s, 64)
		// fmt.Println(p)
		storeVal[idx] = append(storeVal[idx], p)
	}
}
func SkipLines(scanner *bufio.Scanner, lineNum int) {
	for i := 0; i < lineNum; i++ {
		scanner.Scan()
		scanner.Text()
	}
}

func ReadLines(scanner *bufio.Scanner, storeVal []float64, lineNum int) []float64 {
	for i := 0; i < lineNum; i++ {
		scanner.Scan()
		s := scanner.Text()
		p, _ := strconv.ParseFloat(s, 64)
		storeVal = append(storeVal, p)
	}

	return storeVal
}
