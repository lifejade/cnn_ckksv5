package test

import (
	"bufio"
	"context"
	"fmt"
	io "io"
	"sync"
	"time"
)

type slowReader struct {
	mu *sync.Mutex
}

func (reader slowReader) Read(b []byte) (int, error) {
	reader.mu.Lock()
	defer reader.mu.Unlock()
	time.Sleep(time.Second)
	return len(b), nil

}

func test4() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	sr := slowReader{new(sync.Mutex)}
	r := bufio.NewReaderSize(sr, 128)

	for i := 0; i < 10; i++ {
		go func() {

			for {
				buf := make([]byte, 3)
				select {
				case <-ctx.Done():
					return
				default:
				}
				io.ReadFull(r, buf)
				fmt.Println(buf)
			}
		}()
	}

	<-ctx.Done()
}
