The csv files contain the parameters of the points where Earth-Moon halo orbits cross the y=0 plane of the synodic reference frame (the rotating frame of the circular restricted three-body problem), with the absolute value of the z-coordinate being maximum. The first two columns are the x- and z-coordinates of the point, respectively; the third column is the y-component of the velocity at this point. For northern halo orbits, the phase state vector can be constructed as follows: [x 0 z 0 v_y 0]. A twin southern halo orbit differs just in the z-coordinate sign. In the fourth and fifth columns, the orbital period and the Jacobi constant value are listed for each halo orbit. Finally, the sixth column displays the maximum absolute value among the multiplicators (the monodromy matrix eigenvalues). If this parameter is greater than 1, a halo orbit is unstable. The unit parameter indicates linear stability.

The nondimensional systems of units are adopted based on the following data.
---------------------------
Earth-Moon system constants
---------------------------
mu = 1.215058446035100e-02	mass parameter, nondimensional
TU = 4.342564574695797e+00	time unit, days
DU = 3.844050000000000e+05	distance unit, km
VU = 1.024540192302405e+00	velocity unit, km/s

The first rows in the files contain state vectors of those L1/L2 planar Lyapunov orbits that give rise to the corresponding halo orbit family. The last rows contain the parameters of the first collision orbit, an orbit terminating the family.