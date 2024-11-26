#ifndef HDDT_COLL_ALGO_H
#define HDDT_COLL_ALGO_H

#include <coll.h>

/*
 * https://github.com/microsoft/LightGBM/blob/master/src/network/network.cpp
 */
namespace hddt {
/* bruck
 * :for short msg: ALLTOALL (CLASSIC)
 */

/* Double Binary Trees
 * :for short/medium msg: ALLREDUCE (NCCL)
 */

/* Ring
 * :for long msg: ALLREDUCE (NCCL)
 */

/* Recursive Doubling
 * :for short msg: REDUCE-SCATTER (CLASSIC)
 */

/* Pairwise-Exchange
 * :for long msg: REDUCE-SCATTER  (CLASSIC)
 *                ALLTOALL        (CLASSIC)
 */
} // namespace hddt
#endif