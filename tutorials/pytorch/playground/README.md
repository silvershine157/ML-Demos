<h1>Determinsitic Auto Encoder</h1>
<p>Experiment: reconstruction of MNIST image where there is narrow dimension bottleneck</p>
<p>Result: was able to acheive more dimension reduction when we used deeper MLPs as encoder and decoder. (it actually
'learns' something about data rather to just mimic the identity transform). Tuning learning rate was critical.</p>


<h1>State-only RNN</h1>
<p>Experiment: given initial state, predict how it evolves under a fixed, unknown linear dynamical system</p>
<p>Result: scales well for higher dimension, but quickly gets worse for longer sequence</p>
<p>Discussion: this is an instance of long term dependency. Although this is not a chaotic system, gradient explodes
quickly when length is increased</p>
