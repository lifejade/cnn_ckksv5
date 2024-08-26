package cnn

import (
	"fmt"
	"runtime"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"

	"bufio"
	"log"
	"os"
	"strconv"
)

func ResNetCifar10(layerNum int, imageID int) {
	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정

	params, err := hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{
		LogN:            16,
		LogQ:            []int{51, 46, 46, 46, 46},
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

	_, _, _, _ = encryptor, decryptor, encoder, evaluator

	var file *os.File
	//co, st, fh, fw := 0, 0, 3, 3
	init_p := 8
	//logp := 46
	// logp := 40
	//logq := 51
	logn := 15
	n := 1 << logn
	//stage := 0
	//epsilon := 0.00001
	B := 40.0

	// test image
	file, err = os.Open("/home/user/testFile/test_values.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	in := bufio.NewScanner(file)

	SkipLines(in, 32*32*3*imageID)
	var image []float64

	image = ReadLines(in, image, 32*32*3)

	fmt.Println(image)
	for len(image) < n {
		image = append(image, 0.0)
	}
	for i := n / init_p; i < n; i++ { // copy and paste
		image[i] = image[i%(n/init_p)]
	}
	for i := 0; i < n; i++ {
		image[i] /= B
	}
	//fmt.Println(image)
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
