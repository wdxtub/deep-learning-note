// https://leetcode-cn.com/problems/container-with-most-water/

func maxArea(height []int) int {
	left := 0
	right := len(height) -1
	result := 0
	for {
		if left >= right {
			break
		}
		minOne := 0
		gap := right - left
		if height[left] > height[right] {
			minOne = height[right]
			right -= 1
		} else {
			minOne = height[left]
			left += 1
		}
		area := minOne * gap
		if area > result {
			result = area
		}
	}
	return result
}