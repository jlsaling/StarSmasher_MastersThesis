
Before stellar profiles can be used for collisions, stars first have to be imported into StarSmasher and relaxed. Relaxing a star in StarSmasher can be tricky, depending on the stellar structure. From what we understand, relaxation goes as follows:

#### What the code does (in initialize_parent.f)
- StSm loads the stellar profile
- Solves for the necessary temperature profile to give the desired rho and pressure profiles for hydrostatic equilibrium (since StSm uses a different/simplified EOS, hence the temperature profile in StSm will never match the stellar profile one!)
- Creates second derivatives of  the density, internal energy, and mean molecular weight
- Depending on nnopt(!), sets up the initial hcp lattice on which the SPH particles are placed
- Determines if a core particle is needed (condition?)
	- Core particles only interact gravitationally with SPH particles and have no temperature / internal energy
- density, internal energy, and mean molecular weight are assigned to these particles by interpolating the stellar profile values and using the second derivatives computed before
	- In the GitHub version, it uses **linear values and does spline interpolation**. We  (and Jamie Lombardi) modified the code to also have the option to 
		- Interpolate with logarithmic values (helps if StSm interpolates to negative values)
		- Do linear interpolation (instead of using splines)
		- Use clamping on the spline interpolation (untested)
- Things can go wrong here, so it can be worthwhile to experiment with the different interpolation schemes to obtain different initial conditions.
	- Example: We had a case with a 58 Msun PARSEC CHeB star which had irregular shell spacing at the transition from core to envelope. This caused problems with the spline interpolation, namely negative densities and / or internal energies. Using log values fixed this.

#### Setting up a relaxation run

The  most important parameters to consider when relaxing a star are
- n | The total number of particles you want
	- Depends on the  star and the resolution you need. For example, a CHeB star would need more particles than a MS star because of the large density contrast between core and envelope
   	- In the past, we chose ~100000 particles for MS stars, ~900000 particles for a CHeB stars / RGs (nmax is currently 900000, so you have to choose a value slightly below that, otherwise StSm will complain!)
   	- If you want to increase nmax, go into parallel_bleeding_edge/src/starsmasher.h and edit the line at the top where nmax is set.
   	- CAUTION: If you set nmax too high, StarSmasher might not compile. To fix this, go to parallel_bleeding_edge/src/Makefile and add "-mcmodel=medium" to FFLAGS = $(OLEVEL).
- treloff | Time the simulation switches from relaxation to dynamical run
	- From my understanding, for t < treloff, heating from artificial viscosity is turned off. This is because the star naturally undergoes contraction or expansion during the relaxation to find its equilibrium radius, and it is not desirable to have heat generated from that movement
	- You probably have to test how long the star needs to relax, but it might be a good idea to set it to something larger than the dynamical timescale of the star in code units (sqrt(R^3 / M))
- trelax | Timescale for the artificial drag force
	- This is a terrible name for this parameter, but it is very important for properly relaxing stars 
	- During the relaxation, the star can undergo oscillations that can potentially create shocks and blow up the star. To prevent this, an artificial drag force is introduced, which dampens these oscillations/shocks.
	- The lower trelax, the stronger the drag force and the dampening
	- Again, a good initial guess can be sqrt(R^3 / M)

Also important to consider:
- tf | Time to stop the simulation
	- After treloff, the StSm switches to a dynamical calculation, meaning heating from AV is turned on, and the drag force is turned off. It is advisable to set tf such that you can observe the star's behaviour after relaxation to see if everything went well
- nnopt | Parameter governing the number of neighbors
	- This is only really important if you choose to use the WC4 kernel
    - Different nnopt can yield different initial conditions
	- Need to consider:
		- Higher nnopt --> larger smoothing lengths & potentially weaker gravity, runs take longer
		- Too low nnopt --> WC4 kernel is less accurate than cubic spline at low nnopt
        - For collisions, you need the same nnopt for both stars, so if you want to do a larger grid of collisions, be sure to find an nnopt that works for all situations...



#### How do I know a Relaxation is going well

There are a few helpful plots that you can make to see if your star is properly relaxing

1. Density profiles
	- It is always a good idea to compare the StarSmasher profile to the original stellar evolution code profile
	- If it does not match at all at the beginning, try experimenting with the interpolation schemes
	- If it does not match later during/at the end of the relaxation, a shock might have blown up the star --> adjust trelax / treloff (see below)
2. Energy plots
	- StarSmasher stores total, kinetic, internal, and potential energy (among other things) each timestep in energyX.sph files
	- Plot the evolution of the energy
		- E_tot, E_pot, and E_int will likely change very quickly since the initial values are only a "guess" by StSm. However, it is still a good idea to look out for sudden jumps or oscillations.
		- E_kin is an important indicator to see if the particles inside the star are moving (shocks/oscillations). You will likely see oscillations, especially at the beginning of the run
			- If these oscillations cease over time and the star does not blow up, fine!
			- If they continue over time or seem very strong and the star does blow up, look at the period of these oscillations and set trelax to that period (or slightly below that)
3. Map slice radial velocity plots around the orbital plane (z = 0)
	- Very useful indicator to see if and when shocks inside the star develop! 
If everything looks good, take the last output of the run, rename it to sph.start1u or sph.start2u and use it as one of the start files for collision runs.

#### Example
(This section could use some pictures, maybe even add a tutorial with a star known to cause problems, e.g., the 58 Msun PARSEC star)
Let's imagine you want to relax a 58 solar mass CHeB star. This might be a difficult task, since it has a complicated stellar structure. Here is how you could approach this challenge:

- Choose the number of particles
    - For this type of star, ~ 900000 particles might be a good idea (then ~ 100000 for a low-res. run)
- Choose a tf and treloff
    - Set treloff larger than the dynamical timescale of the star
    - Set tf ~ 100 time units larger than that
    - Example: treloff = 400, tf = 500
- Choose trelax
    - For an initial guess, you could either use
     	a) The dynamical timescale of the star or
        b) Do a (short) low-resolution run with a relatively low number of particles, which might already show oscillations in the energy plots. Taking the period of these oscillations could give you a good initial guess for an appropriate trelax.
- Optional, depending on the kernel choice: set nnopt
- Start the run.
	- If the run starts: great!
    - If the run fails: not great, but this can happen. Take a look at log0.sph and see if any errors show up. If not in there, look at the .err and .out files of the Slurm job.
    - Potential errors:
    	- n>nmax: Reduce the number of particles. I have not yet found out how to properly set nmax to something else than 900000....
       	- rho <= 0: Try interpolating in log space
       	- "bad input: imaginary results?" is raised by gettemperature.f and is caused by negative internal energy: Try interpolating in log space
- Plot a density profile of the first output file
		- If the profile does not match the original stellar profile at all, try experimenting with log-space interpolation and/or switch from spline to linear interpolation
- During the relaxation, often do energy plots. If something goes very wrong early in the run (way too strong oscillations, for example), abort the run and start adjusting trelax as described in the sections above.
- To be continued
 
  

TODO: equalmass parameter

