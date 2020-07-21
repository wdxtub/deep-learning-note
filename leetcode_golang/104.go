// https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/

/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
 func maxDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}

	lh := maxDepth(root.Left)
	rh := maxDepth(root.Right)
	if lh > rh {
		return lh + 1
	}
	return rh + 1
}