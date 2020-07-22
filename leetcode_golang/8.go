// https://leetcode-cn.com/problems/string-to-integer-atoi/

func myAtoi(str string) int {
	sign, abs := clean(str)
	result := 0
	for _, c := range abs {
		result = result * 10 + int(c-'0')
		switch {
		case sign == 1 && result > math.MaxInt32:
			return math.MaxInt32
		case sign == -1 && result >  -1 * math.MinInt32:
			return math.MinInt32
		}
	}
	return sign * result
}

func clean(s string) (sign int, abs string) {
	start := 0
    if len(s) == 0 {
        return 
    }
	for {
		if start < len(s) && s[start] == ' '{
			start += 1
		} else {
			break
		}
	}
    if start == len(s) {
        return
    }
	switch s[start] {
	case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
		sign, abs = 1, s[start:]
	case '+':
		sign, abs = 1, s[start+1:]
	case '-':
		sign, abs = -1, s[start+1:]
	default:
		abs = ""
		return
	}
	for idx, c := range abs {
		if c < '0' || '9' < c {
			abs = abs[:idx]
			break
		}
	}
	return
}
