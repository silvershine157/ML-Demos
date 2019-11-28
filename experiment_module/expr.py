import os

class Experiment():

	def __init__(self, args=None):
		self.sub_exprs = []
		self.args = args
		self.local_id = self.make_local_id()

	def make_local_id(self):
		return "default_id"

	def add_expr(self, expr):
		self.sub_exprs.append(expr)

	def run(self, log_dir):
		sub_results = []
		for sub_expr in self.sub_exprs:
			sub_dir = os.join(log_dir, self.local_id)
			os.makedirs(sub_dir)
			sub_res = sub_expr.run(sub_dir)
			results.append(sub_res)
		result = self.produce_result(sub_results, log_dir)
		return result

	def produce_result(self, sub_results, log_dir):
		result = None
		# do experiment, write log
		return result


# Example experiment: tune lambda for MLP on MNIST
class L2RegTuningExp(Experiment):

	def __init__(self, min_lmbda, factor, n_lambdas):
		args = {
			"min_lmbda":min_lmbda,
			"factor":factor,
			"n_lambdas": n_lambdas
		}
		super(RegularizationExp, self).__init__(args)
		for n in range(n_lambdas):
			lmbda = min_lmbda * (factor**n)
			sub_expr = MNISTExp(lmbda)
			self.add_expr(sub_expr)

	def make_local_id(self):
		return "L2RegTuning"

	def produce_result(self, sub_results, log_dir):
		result = []
		for sub_res in sub_results:
			lmbda = sub_res["lmbda"]
			test_accuracy = sub_res["test_accruacy"]
			result.append((lmbda, test_accruacy))
		# TODO: write log
		return result


class MNISTExp(Experiment):

	def __init__(self, lmbda):
		args = {
			"lmbda": lmbda
		}
		super(MNISTExp, self).__init__(args)
		# no subexperiments

	def make_local_id(self):
		return "mnist_lmbda_{0}".format(self.args["lmbda"])

	def produce_result(self, sub_results, log_dir):
		result = {}
		lmbda = self.args["lmbda"]

		# TODO: perform experiment with lmbda, write log
		
		test_accuracy = 0.0
		result["lmbda"] = lmbda
		result["test_accuracy"] = test_accuracy
		# TODO: write log
		return result
