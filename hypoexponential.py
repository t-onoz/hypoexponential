import unittest
from collections.abc import Sequence
from typing import Union
import numpy as np
from scipy.stats import rv_continuous
from scipy.linalg import expm


class hypoexpon_gen(rv_continuous):
    def __init__(self, scales: Union[float, Sequence[float]], **kw):
        """hypoexponential distribution (準指数分布）を生成する。
        X1, X2, X3, ...が期待値Θ1, Θ2, Θ3, ...の指数分布に独立に従うとき、
        X=X1+X3+X3+...が従うのが母数Θ=(Θ1, Θ2, Θ3, ...)の準指数分布である。

        :param scales: 元となる指数分布の期待値を一次元配列として与える。
            ※scipyの仕様上、cdfやpdfメソッドの呼び出しに可変数のパラメータを設定できないため、コンストラクタに指定するように実装した。
        """
        super().__init__(a=0, name='hypoexpon')
        lambda_ = 1.0 / np.ravel(scales)
        self._lambda = lambda_
        k = len(lambda_)
        self._alpha = np.zeros((1, k))
        self._alpha[0, 0] = 1
        self._theta = np.zeros((k, k))
        for i in range(k):
            self._theta[i, i] = -lambda_[i]
            if i != k-1:
                self._theta[i, i+1] = lambda_[i]
        self._one = np.ones((k, 1))

    def _pdf(self, x):
        @np.vectorize
        def _pdf_single(x):
            return (-self._alpha @ expm(x * self._theta) @ self._theta @ self._one).item()
        return _pdf_single(x)

    def _cdf(self, x):
        @np.vectorize
        def _cdf_single(x):
            return 1 - (self._alpha @ expm(x * self._theta) @ self._one).item()
        return _cdf_single(x)


class TestHypoExponential(unittest.TestCase):
    from scipy.stats import expon, gamma

    def test_single(self):
        """母数一つの準指数分布は指数分布に等しい"""
        scale = 1.5
        xs = np.linspace(0, scale * 5, 11)
        pdf_expon = self.expon.pdf(xs, scale=scale)
        cdf_expon = self.expon.cdf(xs, scale=scale)
        hypo = hypoexpon_gen(scale)
        pdf_hypo = hypo.pdf(xs)
        cdf_hypo = hypo.cdf(xs)
        np.testing.assert_allclose(pdf_hypo, pdf_expon)
        np.testing.assert_allclose(cdf_hypo, cdf_expon)

    def test_double(self):
        """母数が等しい準指数分布はガンマ分布に一致"""
        scale = [0.75, 0.75]
        xs = np.linspace(0, sum(scale)*5, 11)
        pdf_gamma = self.gamma.pdf(xs, a=2, scale=scale[0])
        cdf_gamma = self.gamma.cdf(xs, a=2, scale=scale[0])
        hypo = hypoexpon_gen(scale)
        pdf_hypo = hypo.pdf(xs)
        cdf_hypo = hypo.cdf(xs)
        np.testing.assert_allclose(pdf_hypo, pdf_gamma)
        np.testing.assert_allclose(cdf_hypo, cdf_gamma)

    def test_desc(self):
        """異なる母数を含む準指数分布の検証。正しい平均や分散を与えるか"""
        rng = np.random.default_rng(0)
        scales = rng.uniform(0.0, 2.0, 3)
        mean = np.sum(scales)
        var = np.sum(scales**2)
        hypo = hypoexpon_gen(scales)
        self.assertAlmostEqual(hypo.mean(), mean)
        self.assertAlmostEqual(hypo.var(), var)


if __name__ == '__main__':
    unittest.main(verbosity=2)
