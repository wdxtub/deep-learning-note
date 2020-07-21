// https://leetcode-cn.com/problems/longest-common-prefix/

func longestCommonPrefix(strs []string) string {
	position := 0
	length := len(strs)
    if length == 0 {
        return ""
    }
	for {
        if len(strs[0]) == 0 {
            return ""
        }
        if position >= len(strs[0]) {
            break
        }
		char := strs[0][position]
		for i := 1; i < length; i++ {
            if len(strs[i]) == 0 {
                return ""
            }
			if position >= len(strs[i]) || strs[i][position] != char {
				return strs[i][:position]
			}
		}
		position += 1	
	}
	return strs[0][:position]
}