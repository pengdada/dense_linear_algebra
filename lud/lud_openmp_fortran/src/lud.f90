! =================================================================================================
! This file is part of the CodeVault project. The project is licensed under Apache Version 2.0.
! CodeVault is part of the EU-project PRACE-4IP (WP7.3.C).
!
! Author(s):
!   Rafal Gandecki <rafal.gandecki@pwr.edu.pl>
!
! This example demonstrates the use of Fortran 90 with OpenMP for LU decomposition (Doolittle algorithm).
!
! See [http://www.openmp.org/] for the full OpenMP documentation.
!
! =================================================================================================


program lud_openmp

  integer, parameter :: n = 3000
  real, allocatable :: A(:)
  real, allocatable :: L(:)
  real, allocatable :: U(:)
  double precision :: t_config
  integer :: t1, t2, clock_rate, clock_max

  allocate(A(n*n))
  allocate(L(n*n))
  allocate(U(n*n))

  call fill_random(A, n, n)

  call system_clock (t1, clock_rate, clock_max )
  call lud_algorithm(A, L, U, n)
  call system_clock (t2, clock_rate, clock_max )
  t_config = real ( t2 - t1 ) / real ( clock_rate )
  print '("Time without OpenMp: ",f6.3," seconds.")', t_config

  call system_clock (t1, clock_rate, clock_max )
  call lud_openmp_algorithm(A, L, U, n)
  call system_clock (t2, clock_rate, clock_max )
  t_config = real ( t2 - t1 ) / real ( clock_rate )

  print '("Time with OpenMp: ",f6.3," seconds.")', t_config

  deallocate(A)
  deallocate(L)
  deallocate(U)

end program lud_openmp


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


subroutine lud_algorithm(A, L, U, n)

  integer, intent(in) :: n
  real, dimension(n*n), intent(in) :: A
  real, dimension(n*n), intent(inout) :: L
  real, dimension(n*n), intent(inout) :: U
  integer :: i, j, k

  do i=1, n
    do j=1, n
      if (j>i) then
        U((j-1)*n+i) = 0
      end if
      U((i-1)*n+j) = A((i-1)*n+j)
      do k=1, i-1
        U((i-1)*n+j) = U((i-1)*n+j) - (U((k-1)*n+j) * L((i-1)*n+k))
      end do
    end do
    do j=1, n
      if(i>j) then
        L((j-1)*n+i) = 0
      else if (j==i) then
        L((j-1)*n+i) = 1
      else
        L((j-1)*n+i) = A((j-1)*n+i) / U((i-1)*n+i)
        do k=1, i-1
          L((j-1)*n+i) = L((j-1)*n+i) - ((U((k-1)*n+i) * L((j-1)*n+k)) / U((i-1)*n+i))
        end do
      end if
    end do
  end do

end subroutine


subroutine lud_openmp_algorithm(A, L, U, n)
  
  integer, intent(in) :: n
  real, dimension(n*n), intent(in) :: A
  real, dimension(n*n), intent(inout) :: L
  real, dimension(n*n), intent(inout) :: U
  integer :: i, j, k


!$OMP PARALLEL DO DEFAULT (SHARED) PRIVATE(i,j,k)
  do i=1, n
    do j=1, n
      if (j>i) then
        U((j-1)*n+i) = 0
      end if
      U((i-1)*n+j) = A((i-1)*n+j)
      do k=1, i-1
        U((i-1)*n+j) = U((i-1)*n+j) - (U((k-1)*n+j) * L((i-1)*n+k))
      end do
    end do

    do j=1, n
      if(i>j) then
        L((j-1)*n+i) = 0
      else if (j==i) then
        L((j-1)*n+i) = 1
      else
        L((j-1)*n+i) = A((j-1)*n+i) / U((i-1)*n+i)
        do k=1, i-1
          L((j-1)*n+i) = L((j-1)*n+i) - ((U((k-1)*n+i) * L((j-1)*n+k)) / U((i-1)*n+i))
        end do
      end if
    end do
  end do

!$OMP END PARALLEL DO
end subroutine


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
