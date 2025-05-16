use broadcast to send out matrix size
use scatter to send A
use broadcast to send B (need to transpose before we can use scatter)
use MPI_Probe and MPI_Recv with any source to process results

######################
Todo:

1. use gather to retrieve the results
2. transpose B and then use scatter to distribute
