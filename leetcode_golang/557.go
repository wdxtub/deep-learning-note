// https://leetcode-cn.com/problems/reverse-words-in-a-string-iii/

func reverseWords(s string) string {
	if len(s) == 0 {
		return s
	}
	newS := []byte(s)
	i, j := 0, 0
	for ; i < len(newS); {
		for ; i < len(newS) && newS[i] == ' '; i++ {
		}
		for j = i; j < len(newS) && newS[j] != ' '; j++ {			
		}
		rev(newS, i, j-1)
		i = j
	}
	return string(newS)
}

//in place reverse
func rev(ss []byte, l, h int) {
	for l < h {
		ss[l], ss[h] = ss[h], ss[l]
		l++
		h--
	}
}

