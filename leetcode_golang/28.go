// https://leetcode-cn.com/problems/implement-strstr/

func strStr(haystack string, needle string) int {
	l2 := len(needle)
	if l2 == 0 {
		return 0
	}
	l1 := len(haystack)
	for i := 0; i < l1-l2+1; i++ {
		matched := true
		for j := 0; j < l2; j++ {
			if haystack[i+j] != needle[j] {
				matched = false
				break
			}
		}
		if matched {
			return i
		}
	}
	return -1
}	