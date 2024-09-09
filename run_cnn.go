package main

import (
	"fmt"
	"sync"

	"os"
	"strconv"

	"github.com/lifejade/cnn_ckksv5/cnn"
)

func main() {
	//go run run_cnn.go 20 0 2; go run run_cnn.go 20 3 5; go run run_cnn.go 20 6 8; go run run_cnn.go 20 9 11
	var layerNum, startImageID, endImageID int
	if len(os.Args) < 2 {
		fmt.Print("Layer Num(20) : ")
		fmt.Scan(&layerNum)
		fmt.Print("Start image ID: ")
		fmt.Scanln(&startImageID)
		fmt.Print("End image ID: ")
		fmt.Scanln(&endImageID)
	} else if len(os.Args) == 4 {
		var err error
		if layerNum, err = strconv.Atoi(os.Args[1]); err != nil {
			fmt.Println("layer num error")
			os.Exit(0)
		}
		if startImageID, err = strconv.Atoi(os.Args[2]); err != nil {
			fmt.Println("start image number num error")
			os.Exit(0)
		}
		if endImageID, err = strconv.Atoi(os.Args[3]); err != nil {
			fmt.Println("end image number num error")
			os.Exit(0)
		}

		fmt.Printf("Layer Num(20) : %d\n", layerNum)
		fmt.Printf("Start image ID: %d\n", startImageID)
		fmt.Printf("End image ID: %d\n", endImageID)
	} else {
		fmt.Println("args error")
		os.Exit(0)
	}

	var wg sync.WaitGroup
	n := endImageID - startImageID + 1
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			defer wg.Done()
			cnn.ResNetCifar10(layerNum, i+startImageID)
		}()
	}

	wg.Wait()

}
