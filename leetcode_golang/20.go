// https://leetcode-cn.com/problems/valid-parentheses/

func isValid(s string) bool {
	fake_stack := []string{}
	left := "([{"
	// right := ")]}"

	for i := 0; i < len(s); i++ {
		si := string(s[i])
		if strings.Contains(left, si) {
			fake_stack = append(fake_stack, si)
		} else {
			if len(fake_stack) == 0 || 
			(si == ")"  && fake_stack[len(fake_stack)-1] != "(") ||
			(si == "]"  && fake_stack[len(fake_stack)-1] != "[") ||
			(si == "}"  && fake_stack[len(fake_stack)-1] != "{") {
				return false
			} else {
				fake_stack = fake_stack[:len(fake_stack)-1]
			}
			
		}
	}
	if len(fake_stack) != 0 {
		return false
	}

	return true
}