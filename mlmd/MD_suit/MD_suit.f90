module MD_suit

implicit none

contains

	subroutine gausdist(nat,vel_in)
		implicit none
		integer, intent(in) :: nat
		real(8), dimension(3,nat), intent(out) :: vel_in
		!temporal storage
		real:: s1,s2
		real(8):: t1,t2,tt
		real(8),parameter:: eps=1.d-8
		real(8),dimension(3*nat)::  vxyz
		integer:: i,j
		do i=1,3*nat-1,2
			call random_number(s1)
			t1=eps+(1.d0-2.d0*eps)*dble(s1)
			call random_number(s2)
			t2=dble(s2)
			tt=sqrt(-2.d0*log(t1))
			vxyz(i)=tt*cos(6.28318530717958648d0*t2)
			vxyz(i+1)=tt*sin(6.28318530717958648d0*t2)
		enddo
		call random_number(s1)
		t1=eps+(1.d0-2.d0*eps)*dble(s1)
		call random_number(s2)
		t2=dble(s2)
		tt=sqrt(-2.d0*log(t1))
		vxyz(3*nat)=tt*cos(6.28318530717958648d0*t2)
		do j=1, nat
			do i=1, 3
				vel_in(i,j)= vxyz(3*(j-1) + i)
			end do
		end do
		return
	end subroutine gausdist
    
    subroutine init_vel_atoms(amu,temp, nat, vel_out)!, vel_lat_in_bug)
		implicit none
		!*********************************************************************************************
		integer, intent(in) :: nat
		!integer, intent(in) :: nsoften
		real(8), intent(in) :: temp
		!real(8), intent(in) :: bmass
		real(8), intent(in) :: amu(nat)
		!real(8):: amass(nat)
		real(8),dimension(3,nat), intent(out) :: vel_out
		real(8),dimension(3,nat) :: vel_to_scale
		real(8),dimension(3,nat) :: vel_scaled
		real(8),dimension(3) :: p_total
		integer:: iat,i_dim
		real(8), parameter :: Ha_eV=27.21138386d0 ! 1 Hartree, in eV
		real(8), parameter :: kb_evK=8.617343d-5! Boltzmann constant in eV/K
		real(8):: rescale_vel, E_kin

		!Get random Gaussian distributed atomic velocities
		call gausdist(nat,vel_to_scale)
		!scale volocities to the right temperature
		E_kin= 0.d0
		do iat=1,nat
			do i_dim=1,3
				E_kin=E_kin+(vel_to_scale(i_dim,iat)*vel_to_scale(i_dim,iat)*amu(iat))/2.d0
			end do
		end do
		!Now rescale the velocities to give the exact temperature
		rescale_vel=sqrt(3.d0*(nat-1.d0)*kb_evK*temp/(2.d0*E_kin))
		vel_scaled(:,:)=vel_to_scale(:,:)*rescale_vel
		!get rid of center of mass momentum
		p_total= 0.d0
		do iat=1, nat
			p_total= p_total + amu(iat)*vel_scaled(:,iat)
		end do
		p_total= p_total/nat
		do iat=1, nat
			vel_scaled(:,iat) = vel_scaled(:,iat) - p_total(:)/amu(iat)
		end do
		vel_out= vel_scaled
	end subroutine init_vel_atoms
		
	subroutine get_v_mat(amu, v_in, vol, nat, v_mat)
		implicit none
		integer, intent(in) :: nat
		real(8), intent(in) :: v_in(3,nat)
		real(8), intent(in) :: amu(nat)
		real(8), intent(in) :: vol
		real(8), intent(out) :: v_mat(3,3)
		integer :: iat, i, j
		v_mat=0.d0
		do iat=1,nat
			do i=1,3
	   			do j=1,3
			  		v_mat(i,j)= v_mat(i,j)+amu(iat)*v_in(i,iat)*v_in(j,iat)/vol
		   		enddo
			enddo
		enddo
		return
	end subroutine get_v_mat
	
	subroutine get_asv(v_in, amu, nat, asv) 
	!velocity dependent part of the aceleration over the 
	!thermostat degree of freedom
		implicit none
		integer, intent(in) :: nat
		real(8), intent(in) :: v_in(3,nat)
		real(8) :: temp(nat)
		real(8), intent(in) :: amu(nat)
		real(8), intent(out) :: asv
        temp= sum(v_in*v_in, dim=1)
        asv= sum(amu*temp)
		return
	end subroutine get_asv	
	
	
	subroutine get_af_t(fcart_in, amu, nat, af_t)
		implicit none
		integer, intent(in) :: nat
		real(8), intent(in) :: fcart_in(3,nat)
		real(8), intent(in) :: amu(nat)
		real(8), intent(out) :: af_t(3,nat)
		integer :: iat
		do iat=1, nat
			af_t(:, iat)= fcart_in(:,iat)/amu(iat)
		enddo
		return
	end subroutine get_af_t

	subroutine md_nvt(r_in,fcart_in,&
						&vel_in, amu, Qmass, dtion_md, temp, s_in,&
						&s_in_dot, correc_steps,&
						&md_steps,nat,&
						&s_out, s_out_dot, r_out, v_out)
		implicit none
		integer, intent(in) :: nat
		integer, intent(in) :: correc_steps !steps for correction alght
		integer, intent(in) :: md_steps !steps for md integration
		!real(8), intent(in) :: latvec_in(3,3)
		real(8), intent(in) :: r_in(3,nat)
		real(8), intent(in) :: fcart_in(3,nat)
		real(8), intent(in) :: vel_in(3,nat)		
		real(8), intent(in) :: amu(nat)
		real(8), intent(in) :: Qmass !mass of the thermostat
		real(8), intent(in) :: dtion_md
		real(8), intent(in) :: temp
		real(8), intent(in) :: s_in !thermostat variable at time t
		real(8), intent(in) :: s_in_dot !thermostat variable at time t+ dt
		real(8),dimension(3,nat), intent(out):: r_out !position at time t
		real(8),dimension(3,nat), intent(out):: v_out !position at time t
		real(8), intent(out) :: s_out !thermostat variable at time t
		real(8), intent(out) :: s_out_dot !thermostat variable at time t+ dt
		!*********************************************************************
		!Variables for my MD part
		!real(8), parameter :: Ha_eV=27.21138386 ! 1 Hartree, in eV
		real(8) :: kb_evK! Boltzmann constant in eV/K
		!real(8), parameter :: amu_emass=1.660538782d-27/9.10938215d-31 ! 1 atomic mass unit, in electronic mass
		real(8):: m(nat) !mass of atoms
		real(8):: vol
		!integer :: nat= 2
		real(8),dimension(3,nat):: r_t !position at time t
		real(8),dimension(3,nat):: r_t_dt !position at time t+ dt
		real(8) :: s_t !thermostat variable at time t
		real(8) :: s_t_dt !thermostat variable at time t+ dt
		real(8) :: s_t_dot !time derivative of thermostat variable at time t
		real(8),dimension(3,nat):: af_t !aceleration force at time t F/m
		real(8),dimension(3,nat):: af_t_dt !aceleration force at time t + dt
		real(8),dimension(3,nat):: ar_t !atomic aceleration at time t
		real(8),dimension(3,nat):: ar_c !atomic aceleration at corrections steps
		real(8) :: asvt ! aceleration dependent on the velocity over the thermostat at time t
		real(8) :: asvc ! aceleration dependent on the velocity over the thermostat at correction step
		real(8) :: as_t ! aceleration over the thermostat at time t
		real(8) :: as_c ! aceleration over the thermostat at correction steps
		real(8),dimension(3,nat):: v_t !velocity at time t
		real(8),dimension(3,nat):: v_c !velocity at correction steps
		real(8) :: s_c_dot !time derivative of thermostat variable at correction steps
		real(8) :: ndf ! nomber of degrees of freesom
		


		real(8):: dt
		integer:: md_i, correc_step_i, iat
		!Assign masses to each atom (for MD)

		!start MD parameters
		kb_evK=(8.617343d-5)
		r_t= r_in!matmul(latvec_in, xred_in)
		v_t= vel_in
		s_t= s_in
		s_t_dot= s_in_dot
		ndf= 3.d0*nat
		dt=dtion_md !define time step
		call get_af_t(fcart_in, amu, nat, af_t)
		call get_asv(v_t, amu, nat, asvt) 
        do md_i= 1, md_steps
			ar_t= (af_t/(s_t**2.d0) - 2.d0*(s_t_dot/s_t**2.d0)*v_t)
			as_t= ((asvt/s_t - (ndf+1)*kb_evK*temp/s_t)/Qmass)
			r_t_dt= r_t + dt*(v_t/s_t) + 0.d5*dt*dt*ar_t
			s_t_dt= s_t + dt*s_t_dot + 0.d5*dt*dt*as_t
			!write(*,*) 'in the md part'
			!prediction correction 
			!prediction, calculate input for correction data
			v_c= s_t_dt*v_t/s_t + dt*s_t_dt*ar_t
			s_c_dot= s_t_dot + dt*as_t
			call get_asv(v_c, amu, nat, asvc)
			
			!correction steps
			do correc_step_i=1, correc_steps
				ar_c= af_t/(s_t_dt**2.d0) - 2.d0*(s_c_dot/s_t_dt**2.d0)*v_c
				as_c= (asvc/s_t_dt - (ndf+1)*kb_evK*temp/s_t_dt)/Qmass    
				v_c= v_t + 0.d5*dt*(ar_c + ar_t)
				s_c_dot= s_t_dot + 0.d5*dt*(as_c + as_t)
				call get_asv(v_c, amu, nat, asvc) 
			end do !correction
			r_t= r_t_dt
			s_t= s_t_dt
			v_t= v_c
			s_t_dot= s_c_dot
		end do !md steps
		s_out= s_t
		s_out_dot= s_t_dot
		!call get_x_t(latvec_in, r_t, nat, x_out)
		r_out= r_t
		v_out= v_t
		return
	end subroutine md_nvt
end module MD_suit
