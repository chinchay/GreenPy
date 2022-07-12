program algebra
    implicit none
    integer :: i, j, size
    real(8) :: energy, delta
    complex(8) :: invE
    complex(8), allocatable, dimension(:,:) :: eye, g, t00, t, td, r_solve
    integer :: time_start, time_end, nTimes
    integer, parameter :: nSizes = 3 !14
    integer, dimension(nSizes) :: size_list
    
    energy  = -2.0
    delta   = 0.01
    invE    = 1 / dcmplx( energy, delta )

    ! size_list = (/ 5, 10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 /)
    ! size_list = (/ 10, 1000, 2000, 4000 /)
    size_list = (/ 5, 10, 20 /)

    do j = 1, nSizes
      size = size_list(j)
      eye  = getEye(size)
      g    = invE * eye
      t00  = eye
      t    = eye
      td   = eye

      time_start = getTime()
      nTimes = 10 !100
      do i = 1, nTimes
        r_solve = renormalize(g, g, size) ! just a toy example
      enddo
      time_end = getTime()
      call displayTimeElapsed(time_start, time_end, nTimes)
    enddo


    contains

    function renormalize(zetas, q, n) result(r)
        implicit none
        !use Routines1    
        !! calcula:
        !! GR=  Q = inv(  Ident - Zetas  )*Q
        !!
        integer,    intent(in) :: n
        complex(8), dimension(n, n), intent(in)  :: zetas, q
        complex(8), dimension(n, n) :: temp1, temp2, eye, r
    
        eye = getEye(n)
        temp1 = eye - zetas

        ! call cinv( temp1, n, n, temp2)   ! devuelve `temp2`
        ! r = matmul(temp2, q) 

        r = q ! r will be overwritten with the solution
        call solve(temp1, r, n)     ! Solve the system temp1 * X = r, overwriting r with X

    end function

    subroutine solve(a, b, size)
      implicit none
      ! DGESV computes the solution to system of linear equations A * X = B for GE matrices
      ! Solve the system A*X = B, overwriting B with X.
      integer, intent(in) :: size
      integer :: n, nrhs, lda, ldb, info
      complex(8), dimension(size, size) :: a
      complex(8), dimension(size, size) :: b
      integer, dimension(size) :: ipiv
      
      n    = size
      nrhs = size
      lda  = n
      ldb  = n
      call dgesv(n, nrhs, a, lda, ipiv, b, ldb, info)
    end subroutine

      function getTime() result(current_milisecond)
        implicit none
        integer, dimension(8) :: values
        integer :: current_milisecond, min2milisec, sec2milisec
        call date_and_time(VALUES=values)
        min2milisec = values(6) * 1000 * 60
        sec2milisec = values(7) * 1000 
        current_milisecond = min2milisec + sec2milisec + values(8)
    end function

    subroutine displayTimeElapsed(time_start, time_end, nTimes)
      implicit none
      integer, intent(in) :: time_start, time_end, nTimes
      real :: delta
      delta = 1.0 * (time_end - time_start) / nTimes
      ! if (delta < 1) then
      !   delta = delta * 1000
      !   write(*, 100) delta, " u-seconds"
      ! else
      !   write(*, 100) delta, " m-seconds"
      ! endif
      ! 100 format (F10.1, A)
      write(*, *) delta
    end subroutine

    function getEye(size) result(eye)
        implicit none
        integer, intent(in) :: size
        complex(8), dimension(size, size) :: eye
        integer :: i
        eye = 0.0
        do i = 1, size
          eye(i, i) = 1.0
        enddo
    end function

    ! Compute inverse of complex matrix
    !http://www.algarcia.org/nummeth/Fortran/cinv.f
    ! Compute inverse of complex matrix

    SUBROUTINE cinv( A, N, MAXN, Ainv )
        implicit none
        integer*4 N, MAXN
        complex(8) A(MAXN,MAXN), Ainv(MAXN,MAXN)
          ! Inputs
          !   A       Matrix A to be inverted
          !   N       Elements used in matrix A (N by N)
          !  MAXN     Matrix dimenstions as A(MAXN,MAXN)
          ! Outputs
          !  Ainv     Inverse of matrix A

        integer*4 MAXMAXN
        parameter( MAXMAXN = 200 )
        integer*4 i, j, k, index(MAXMAXN), jPivot, indexJ
        real(8) scale(MAXMAXN), scaleMax, ratio, ratioMax
        complex(8) AA(MAXMAXN,MAXMAXN), B(MAXMAXN,MAXMAXN), coeff, sum


        if( MAXN .gt. MAXMAXN ) then
          write(*,*) 'ERROR in cinv: Matrix too large'
          stop
        endif

        !* Matrix B is initialized to the identity matrix
        do i=1,N
          do j=1,N
            AA(i,j) = A(i,j)  ! Copy matrix so as not to overwrite
            B(i,j) = 0.0d0
          enddo
          B(i,i) = 1.0d0
        enddo

        !* Set scale factor, scale(i) = max( |a(i,j)| ), for each row
        do i=1,N
          index(i) = i     ! Initialize row index list
          scaleMax = 0.0d0
          do j=1,N
            if( abs(AA(i,j)) .gt. scaleMax ) then
              scaleMax = abs(AA(i,j))
            endif
          enddo
          scale(i) = scaleMax
        enddo

        !* Loop over rows k = 1, ..., (N-1)
        do k=1,(N-1)
          !* Select pivot row from max( |a(j,k)/s(j)| )
          ratiomax = 0.0d0
          jPivot = k
          do i=k,N
            ratio = abs(AA(index(i),k))/scale(index(i))
            if( ratio .gt. ratiomax ) then
              jPivot=i
              ratiomax = ratio
            endif
          enddo
          !* Perform pivoting using row index list
          indexJ = index(k)
          if( jPivot .ne. k ) then     ! Pivot
            indexJ = index(jPivot)
            index(jPivot) = index(k)   ! Swap index jPivot and k
            index(k) = indexJ
          endif
          !* Perform forward elimination
          do i=k+1,N
            coeff = AA(index(i),k)/AA(indexJ,k)
            do j=k+1,N
              AA(index(i),j) = AA(index(i),j) - coeff*AA(indexJ,j)
            enddo
            AA(index(i),k) = coeff
            do j=1,N
              B(index(i),j) = B(index(i),j) - AA(index(i),k)*B(indexJ,j)
            enddo
          enddo
        enddo

        !* Perform backsubstitution
        do k=1,N
          Ainv(N,k) = B(index(N),k)/AA(index(N),N)
          do i=N-1,1,-1
            sum = B(index(i),k)
            do j=i+1,N
              sum = sum - AA(index(i),j)*Ainv(j,k)
            enddo
            Ainv(i,k) = sum/AA(index(i),i)
          enddo
        enddo

    END SUBROUTINE cinv
      
end program algebra
