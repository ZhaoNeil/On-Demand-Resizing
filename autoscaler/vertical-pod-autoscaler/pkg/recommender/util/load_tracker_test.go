package util

import (
	"testing"
)

func TestAverage(t *testing.T) {
	test := NewSimpleMovingAverage(10)
	if test.average != 0 {
		t.Error("error")
	}
	test.Add(1)
	if test.average != 1 {
		t.Fail()
	}
}
