// https://leetcode-cn.com/problems/search-a-2d-matrix-ii/

func searchMatrix(matrix [][]int, target int) bool {
	if len(matrix) == 0 {
		return false
	}
	colIndex := 0
	rowIndex := len(matrix) - 1
	width := len(matrix[0])
	for rowIndex >= 0 && col < width {
		if matrix[rowIndex][colIndex] == target {
			return true
		} else if matrix[rowIndex][colIndex] < target {
			colIndex++
		} else {
			rowIndex--
		}
	}
	return false
}