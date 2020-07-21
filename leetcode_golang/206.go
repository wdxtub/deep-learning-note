// https://leetcode-cn.com/problems/reverse-linked-list/

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func reverseList(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}
	
	var prev *ListNode
	var next *ListNode
	prev = nil
	for {
		if head.Next == nil{
			break
		}
		next = head.Next
		head.Next = prev

		prev = head
		head = next
	}
	return head
}