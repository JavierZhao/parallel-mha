#ifndef MULTIHEAD_ATTENTION_H
#define MULTIHEAD_ATTENTION_H

#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

class MultiHeadAttention {
public:
    MultiHeadAttention(int num_heads, int d_model);

    MatrixXd compute(const MatrixXd& x);

    void set_weights(const MatrixXd& Wq, const MatrixXd& Wk, const MatrixXd& Wv, const MatrixXd& Wo);

private:
    int num_heads;
    int d_model;
    int depth;

    std::vector<MatrixXd> Wq;
    std::vector<MatrixXd> Wk;
    std::vector<MatrixXd> Wv;
    MatrixXd Wo;

    std::vector<MatrixXd> split_heads(const MatrixXd& x);
    MatrixXd scaled_dot_product_attention(const MatrixXd& Q, const MatrixXd& K, const MatrixXd& V);
    MatrixXd softmax(const MatrixXd& x);
};

#endif // MULTIHEAD_ATTENTION_H
