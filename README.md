Welcome to the git repo for Raf's setup_1D_neutrals_mgop module! (An extension for the jax-sn package by Timo Bogaarts)

The purpose of this extension is to solve the Boltzmann equation for arbitrary neutrals instead of just neutrons.
This is done by rewriting the neutrals equation into the same form as the one used by jax-sn.
With this rewritten equation, we get a set of equations that map the two Boltzmann equations to one another
The goal of this module is to implement these mappings into the code.

For the sake of simplicity, this model only considers a 1D domain (hence the name).
Higher dimensional extensions of this model should be possible, but have not been implemented as of yet.

NOTE: Due to time constrains, this model does not yet include the scattering source term of the Boltzmann equation.
As such, the results produced by this model are not fully physical, and should thus not be regarded as such.
The other terms of the equation (the total scattering area and the external source term) have been tested
and seem to be operating correctly.

Further extension of this module should firstly aim towards implementing the scattering source term,
after which extension into higher-dimensional domains may be considered.

Lastly, this code has been made with substantial direct and indirect help from Timo Bogaarts; so thank you, Timo.

- R.K.K. van der Veeken
