// https://leetcode-cn.com/problems/multiply-strings/

func multiply(num1 string, num2 string) string {
	l1, l2 := len(num1), len(num2)
	
	if l1 == 0 || l2 == 0 {
		return ""
	}
	if (l1 == 1 && num1[0] == '0') || (l2 == 1 && num2[0] == '0') {
		return "0"
	}
	

}