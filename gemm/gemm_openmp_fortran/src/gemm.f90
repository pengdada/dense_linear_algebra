
! =================================================================================================
! This file is part of the CodeVault project. The project is licensed under Apache Version 2.0.
! CodeVault is part of the EU-project PRACE-4IP (WP7.3.C).
!
! Author(s):
!   Mariusz Uchronski <mariusz.uchronski@pwr.edu.pl>
!
! This example demonstrates the use of Fortran 90 with OpenMP for matrix-matrix multiplication.
! The example is set-up to perform single precision matrix-matrix multiplication.
!
! See [http://openmp.org] for the full OpenMP documentation.
!
! =================================================================================================

program gemm_openmp

  implicit none
  integer,parameter :: seed = 86456
  integer, parameter :: n = 400
  integer, parameter :: m = 600
  integer, parameter :: k = 800
  real, allocatable :: a(:)
  real, allocatable :: b(:)
  real, allocatable :: c(:)

  allocate(a(n*m))
  allocate(b(m*k))
  allocate(c(n*k))

  call fill_random(a, n, m)
  call fill_random(b, m, k)

  call run_gemm_openmp(a, b, c, n, m, k)

  deallocate(a)
  deallocate(b)
  deallocate(c)


end program gemm_openmp


subroutine fill_random(a, n, m)

  implicit none

  integer, intent(in) :: n, m
  real, dimension(n*m), intent(inout) :: a
  integer :: i, j
  do i=1, n
    do j=1, m
      a((i-1)*m+j) = rand()
    end do
  end do

end subroutine fill_random

subroutine print_array(a, n, m)

  implicit none

  integer, intent(in) :: n, m
  real, dimension(n*m), intent(in) :: a
  integer :: i, j
  do i=1, n
    do j=1, m
      print '(F8.4,$)', a((i-1)*m+j)
    end do
    print *, ''
  end do

end subroutine print_array


subroutine run_gemm_openmp(a, b, c, n, m, k)

  integer, intent(in) :: n, m , k
  real, dimension(n*m), intent(in) :: a
  real, dimension(m*k), intent(in) :: b
  real, dimension(n*k), intent(inout) :: c
  integer :: i, j, l
  real :: summ


!$OMP PARALLEL DO DEFAULT (NONE) &
!$OMP SHARED(n,m,k,a,b,c) PRIVATE(i,j,l,summ)
  do i=1, n
    do j=1, k
      summ = 0
      do l=1, m
        summ = summ + a((i-1)*m+l) * b((l-1)*k+j)
      end do
      c((i-1)*k+j) = summ
    end do
  end do
!$OMP END PARALLEL DO

end subroutine
