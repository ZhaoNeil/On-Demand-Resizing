package util

type SimpleMovingAverage struct {
	windowsize int
	values     []float64
	sum        float64
	average    float64
}

func NewSimpleMovingAverage(n int) *SimpleMovingAverage {
	return &SimpleMovingAverage{
		windowsize: n,
		values:     make([]float64, n),
	}
}

func (list *SimpleMovingAverage) Add(val float64) float64 {
	if len(list.values) < list.windowsize {
		list.sum += val
		list.values = append(list.values, val)
		list.average = float64(list.sum) / float64(len(list.values))
	} else if len(list.values) == list.windowsize {
		var first_item float64
		first_item, list.values = list.values[0], list.values[1:]
		list.sum -= first_item
		list.values = append(list.values, val)
		list.sum += val
		list.average = float64(list.sum) / float64(len(list.values))
	}
	return list.average
}

func (list *SimpleMovingAverage) Average() float64 {
	return list.average
}
