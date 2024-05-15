#include "multihead_attention.h"
#include <cmath>
#include <iostream>

MultiHeadAttention::MultiHeadAttention(int num_heads, int d_model)
    : num_heads(num_heads), d_model(d_model), depth(d_model / num_heads) {
    Wq.resize(num_heads);
    Wk.resize(num_heads);
    Wv.resize(num_heads);

    for (int i = 0; i < num_heads; ++i) {
        Wq[i] = MatrixXd::Random(d_model, depth);
        Wk[i] = MatrixXd::Random(d_model, depth);
        Wv[i] = MatrixXd::Random(d_model, depth);
    }

    Wo = MatrixXd::Random(d_model, d_model);
}

void MultiHeadAttention::set_weights(const MatrixXd& Wq_in, const MatrixXd& Wk_in, const MatrixXd& Wv_in, const MatrixXd& Wo_in) {
    for (int i = 0; i < num_heads; ++i) {
        Wq[i] = Wq_in.block(0, i * depth, d_model, depth);
        Wk[i] = Wk_in.block(0, i * depth, d_model, depth);
        Wv[i] = Wv_in.block(0, i * depth, d_model, depth);
    }
    Wo = Wo_in;
}

std::vector<MatrixXd> MultiHeadAttention::split_heads(const MatrixXd& x) {
    std::vector<MatrixXd> heads(num_heads);
    int batch_size = x.rows();

    for (int i = 0; i < num_heads; ++i) {
        heads[i] = x.block(0, i * depth, batch_size, depth);
    }

    return heads;
}

MatrixXd MultiHeadAttention::scaled_dot_product_attention(const MatrixXd& Q, const MatrixXd& K, const MatrixXd& V) {
    MatrixXd matmul_qk = Q * K.transpose();
    double dk = static_cast<double>(K.cols());
    MatrixXd scaled_attention_logits = matmul_qk / std::sqrt(dk);

    MatrixXd attention_weights = softmax(scaled_attention_logits);

    return attention_weights * V;
}

MatrixXd MultiHeadAttention::softmax(const MatrixXd& x) {
    MatrixXd exp_x = x.array().exp();
    return exp_x.array().colwise() / exp_x.rowwise().sum().array();
}

MatrixXd MultiHeadAttention::compute(const MatrixXd& x) {
    int batch_size = x.rows();

    std::vector<MatrixXd> Q_heads(num_heads);
    std::vector<MatrixXd> K_heads(num_heads);
    std::vector<MatrixXd> V_heads(num_heads);

    for (int i = 0; i < num_heads; ++i) {
        Q_heads[i] = x * Wq[i];
        K_heads[i] = x * Wk[i];
        V_heads[i] = x * Wv[i];
    }

    std::vector<MatrixXd> attention_heads(num_heads);
    for (int i = 0; i < num_heads; ++i) {
        attention_heads[i] = scaled_dot_product_attention(Q_heads[i], K_heads[i], V_heads[i]);
    }

    MatrixXd concat_attention(batch_size, num_heads * depth);
    for (int i = 0; i < num_heads; ++i) {
        concat_attention.middleCols(i * depth, depth) = attention_heads[i];
    }

    return concat_attention * Wo;
}
