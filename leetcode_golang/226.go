// https://leetcode-cn.com/problems/invert-binary-tree/

/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
 func invertTree(root *TreeNode) *TreeNode {
    if root == nil {
        return nil
    }
	root.Left, root.Right = invertTree(root.Right), invertTree(root.Left)
	return root
}