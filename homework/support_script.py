from scipy.sparse import coo_array, hstack
A = coo_array([[1, 2], [3, 4]])
B = coo_array([[5], [6]])
new_arr = hstack([A, B]).toarray()
pass


l1 = ['the best pizza flour', 'pie making with sweet rice flour', 'good and cheap', 'great treats for the treat ball.', 'the babies love it']
l2 = [word.split() for word in l1]
l2

words = []
for word_list in l2:
    for word in word_list:
        words.append(word)

words_uniq = set(words)
pass

# newaxis is None
# True
# x = np.arange(3)
# x
# array([0, 1, 2])
# x[:, newaxis]
# array([[0],
# [1],
# [2]])
# x[:, newaxis, newaxis]
# array([[[0]],
# [[1]],
# [[2]]])
# x[:, newaxis] * x
# array([[0, 0, 0],
# [0, 1, 2],
# [0, 2, 4]])