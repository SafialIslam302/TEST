#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
  int id,i;
  
  #pragma omp parallel for private(id) /* OpenMP Pragma */
  for (i = 0; i < 8; ++i)
  {
    id = omp_get_thread_num();
    printf("Hello World from thread %d\n", id);
    if (id == 0)
      printf("There are %d threads\n", omp_get_num_threads());
  }
  return 0;
}
