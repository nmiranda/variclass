import theano.tensor as T

coding_dist = T.matrix('coding_dist')
true_dist = T.matrix('true_dist')
#y = T.ivector('y')
loss = T.nnet.binary_crossentropy(coding_dist, true_dist)

print loss.eval({coding_dist: [[0.5,0.2],[0.5,0.2]], true_dist: [[1,0],[1,0]]})
print loss.eval({coding_dist: [[0.2,0.5],[0.2,0.5]], true_dist: [[1,0],[1,0]]})
#print loss.eval({x: [[1,0],[1,0]], y: [1,1]})

#kek = -T.sum(true_dist * T.log(coding_dist), axis=coding_dist.ndim - 1)
#kek = T.log(coding_dist)
#print kek.eval({coding_dist: [[1,0],[1,0]], true_dist: [[1,0],[1,0]]})
#print kek.eval({coding_dist: [[1,0],[1,0]]})