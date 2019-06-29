<h1>State-only RNN</h1>
<p>Experiment: given initial state, predict how it evolves under a fixed, unknown linear dynamical system</p>
<p>Result: scales well for higher dimension, but quickly gets worse for longer sequence</p>
<p>Discussion: this is an instance of long term dependency. Although this is not a chaotic system, gradient explodes
quickly when length is increased</p>
