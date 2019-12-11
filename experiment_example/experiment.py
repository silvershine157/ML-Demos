import os
from dataset import *
from model import *
from train import *
from utils import *

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
		log("Starting experiment: "+log_dir, self)
		sub_results = []
		for sub_expr in self.sub_exprs:
			sub_dir = os.path.join(log_dir, sub_expr.local_id)
			os.makedirs(sub_dir)
			sub_res = sub_expr.run(sub_dir)
			sub_results.append(sub_res)
		result = self.produce_result(sub_results, log_dir)
		log("Exiting experiment: "+log_dir, self)
		return result

	def produce_result(self, sub_results, log_dir):
		result = None
		# do experiment, write log
		return result


class L2RegTuningExp(Experiment):
	def __init__(self, min_lmbda, factor, n_lambdas):
		args = {
			"min_lmbda":min_lmbda,
			"factor":factor,
			"n_lambdas": n_lambdas
		}
		super(L2RegTuningExp, self).__init__(args)
		for n in range(n_lambdas):
			lmbda = min_lmbda * (factor**n)
			# TODO: obtain confidence interval
			sub_expr = BostonMultiExp(lmbda)
			self.add_expr(sub_expr)

	def make_local_id(self):
		return "L2RegTuning"

	def produce_result(self, sub_results, log_dir):
		result = []
		for sub_res in sub_results:
			lmbda = sub_res["lmbda"]
			ci_low = sub_res["ci_low"]
			ci_high = sub_res["ci_high"]
			result.append((lmbda, ci_low, ci_high))
		# TODO: write log
		return result

class BostonMultiExp(Experiment):

	def __init__(self, lmbda):
		args = {
			"lmbda": lmbda
		}
		super(BostonMultiExp, self).__init__(args)
		for trial in range(5):
			sub_expr = BostonExp(lmbda, trial)
			self.add_expr(sub_expr)

	def make_local_id(self):
		return "lmbda_{0}".format(self.args["lmbda"])

	def produce_result(self, sub_results, log_dir):
		perf_list = []
		for sub_res in sub_results:
			perf = sub_res["test_perf"]
			perf_list.append(perf)
		result = {}
		result["lmbda"] = self.args["lmbda"]
		result["ci_low"] = min(perf_list) # TODO: obtain confidence interval
		result["ci_high"] = max(perf_list)
		return result

class BostonExp(Experiment):

	def __init__(self, lmbda, trial):
		args = {
			"trial": trial,
			"lmbda": lmbda,
			"batch_size": 64
		}
		super(BostonExp, self).__init__(args)
		# no subexperiments

	def make_local_id(self):
		return "trial_{0}".format(self.args["trial"])

	def produce_result(self, sub_results, log_dir):
		result = {}
		lmbda = self.args["lmbda"]

		data = prepare_data()
		train_loader, val_loader, test_loader = get_dataloader(data, self.args["batch_size"])
		net = MLP(d_input=data["train_X"].shape[1])
		train_info = train_model(net, train_loader, val_loader, expr=self)
		test_info = test_model(net, test_loader, expr=self)
		
		result["test_perf"] = test_info["test_perf"]
		return result


def test():
	os.system("rm -rf results")
	expr = L2RegTuningExp(min_lmbda=0.1, factor=10., n_lambdas=3)
	expr.run("results")





test()