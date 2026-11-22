//go:build arm64 && !purego

#include "textflag.h"

#define FADD(D,N,M) WORD $(0x4e60d400 | ((M)<<16) | ((N)<<5) | (D))
#define FSUB(D,N,M) WORD $(0x4ee0d400 | ((M)<<16) | ((N)<<5) | (D))
#define FMUL(D,N,M) WORD $(0x6e60dc00 | ((M)<<16) | ((N)<<5) | (D))
#define FDIV(D,N,M) WORD $(0x6e60fc00 | ((M)<<16) | ((N)<<5) | (D))
#define FCMGE(D,N,M) WORD $(0x6e60e400 | ((M)<<16) | ((N)<<5) | (D))
#define FABS(D,N) WORD $(0x4ee0f800 | ((N)<<5) | (D))
#define FCMEQZERO(D,N) WORD $(0x4ee0d800 | ((N)<<5) | (D))

// Strict comparisons select y on equality. Include a left-NaN mask
// to preserve propagation, rather than use FMIN/FMAX's zero tie rule.
#define EXTREMA(CMP,EQ) \
	VLD1.P 16(R1),[V0.B16]; \
	VLD1.P 16(R2),[V1.B16]; \
	WORD $CMP; \
	WORD $EQ; \
	WORD $0x6e205863; \
	VORR V3.B16,V2.B16,V2.B16; \
	VBSL V1.B16,V0.B16,V2.B16; \
	VST1.P [V2.B16],16(R0); \
	SUBS $1,R3,R3

TEXT ·extrema64NEON(SB),NOSPLIT,$0-73
	MOVD dst_base+0(FP),R0
	MOVD x_base+24(FP),R1
	MOVD y_base+48(FP),R2
	MOVD dst_len+8(FP),R3
	LSR $1,R3,R3
	MOVBU maximum+72(FP),R4
	CBNZ R4,max64
min64:
	EXTREMA(0x6ee0e422,0x4e60e403) // fcmgt v2.2d,v1.2d,v0.2d
	BNE min64
	RET
max64:
	EXTREMA(0x6ee1e402,0x4e60e403)
	BNE max64
	RET

TEXT ·extrema32NEON(SB),NOSPLIT,$0-73
	MOVD dst_base+0(FP),R0
	MOVD x_base+24(FP),R1
	MOVD y_base+48(FP),R2
	MOVD dst_len+8(FP),R3
	LSR $2,R3,R3
	MOVBU maximum+72(FP),R4
	CBNZ R4,max32
min32:
	EXTREMA(0x6ea0e422,0x4e20e403)
	BNE min32
	RET
max32:
	EXTREMA(0x6ea1e402,0x4e20e403)
	BNE max32
	RET

#define EXPTERM(C) FMUL(20,20,0); FADD(20,20,C)
TEXT ·exp64NEON(SB),NOSPLIT,$0-56
	MOVD dst_base+0(FP),R0
	MOVD src_base+24(FP),R1
	MOVD dst_len+8(FP),R2
	MOVD constants+48(FP),R3
	LSR $1,R2,R2
	VLD1R.P 8(R3),[V1.D2]
	VLD1R.P 8(R3),[V2.D2]
	VLD1R.P 8(R3),[V3.D2]
	VLD1R.P 8(R3),[V4.D2]
	VLD1R.P 8(R3),[V5.D2]
	VLD1R.P 8(R3),[V6.D2]
	VLD1R.P 8(R3),[V7.D2]
	VLD1R.P 8(R3),[V8.D2]
	VLD1R.P 8(R3),[V9.D2]
	VLD1R.P 8(R3),[V10.D2]
	VLD1R.P 8(R3),[V11.D2]
	VLD1R.P 8(R3),[V12.D2]
	VLD1R.P 8(R3),[V13.D2]
	VLD1R.P 8(R3),[V14.D2]
	VLD1R.P 8(R3),[V15.D2]
	VLD1R.P 8(R3),[V16.D2]
	VLD1R.P 8(R3),[V17.D2]
	VLD1R.P 8(R3),[V18.D2]
	VLD1R (R3),[V19.D2]
expLoop:
	VLD1.P 16(R1),[V0.D2]
	FMUL(21,0,1)
	FADD(21,21,2)
	FSUB(21,21,2)
	FMUL(22,21,4)
	FSUB(0,0,22)
	FMUL(22,21,5)
	FSUB(0,0,22)
	FADD(21,21,3)
	VSHL $52,V21.D2,V21.D2
	VORR V6.B16,V6.B16,V20.B16
	EXPTERM(7)
	EXPTERM(8)
	EXPTERM(9)
	EXPTERM(10)
	EXPTERM(11)
	EXPTERM(12)
	EXPTERM(13)
	EXPTERM(14)
	EXPTERM(15)
	EXPTERM(16)
	EXPTERM(17)
	EXPTERM(18)
	EXPTERM(19)
	FMUL(20,20,21)
	VST1.P [V20.D2],16(R0)
	SUBS $1,R2,R2
	BNE expLoop
	RET

#define A32(D,N,M) WORD $(0x4e20d400 | ((M)<<16) | ((N)<<5) | (D))
#define S32(D,N,M) WORD $(0x4ea0d400 | ((M)<<16) | ((N)<<5) | (D))
#define M32(D,N,M) WORD $(0x6e20dc00 | ((M)<<16) | ((N)<<5) | (D))
#define EXPTERM32(C) M32(20,20,0); A32(20,20,C)
TEXT ·exp32NEON(SB),NOSPLIT,$0-56
	MOVD dst_base+0(FP),R0
	MOVD src_base+24(FP),R1
	MOVD dst_len+8(FP),R2
	MOVD constants+48(FP),R3
	LSR $2,R2,R2
	VLD1R.P 4(R3),[V1.S4]
	VLD1R.P 4(R3),[V2.S4]
	VLD1R.P 4(R3),[V3.S4]
	VLD1R.P 4(R3),[V4.S4]
	VLD1R.P 4(R3),[V5.S4]
	VLD1R.P 4(R3),[V6.S4]
	VLD1R.P 4(R3),[V7.S4]
	VLD1R.P 4(R3),[V8.S4]
	VLD1R.P 4(R3),[V9.S4]
	VLD1R.P 4(R3),[V10.S4]
	VLD1R.P 4(R3),[V11.S4]
	VLD1R.P 4(R3),[V12.S4]
	VLD1R (R3),[V13.S4]
exp32Loop:
	VLD1.P 16(R1),[V0.S4]
	M32(21,0,1)
	A32(21,21,2)
	S32(21,21,2)
	M32(22,21,4)
	S32(0,0,22)
	M32(22,21,5)
	S32(0,0,22)
	A32(21,21,3)
	VSHL $23,V21.S4,V21.S4
	VORR V6.B16,V6.B16,V20.B16
	EXPTERM32(7)
	EXPTERM32(8)
	EXPTERM32(9)
	EXPTERM32(10)
	EXPTERM32(11)
	EXPTERM32(12)
	EXPTERM32(13)
	M32(20,20,21)
	VST1.P [V20.S4],16(R0)
	SUBS $1,R2,R2
	BNE exp32Loop
	RET

// Select A if the mask M is set, otherwise B; M remains available.
#define SELECT(M,A,B,D) WORD $(0x4ea01c00 | ((M)<<16) | ((M)<<5) | D); WORD $(0x6e601c00 | ((B)<<16) | ((A)<<5) | D)

#define COMPLEX_MUL_WIDE \
	FMUL(4,0,2); FMUL(5,1,3); FSUB(4,4,5); \
	FMUL(5,0,3); FMUL(6,1,2); FADD(5,5,6)

#define COMPLEX_DIV_WIDE(STORE) \
	FABS(4,2); FABS(5,3); FCMGE(6,4,5); \
	FCMEQZERO(7,4); FCMEQZERO(13,5); VAND V13.B16,V7.B16,V7.B16; \
	SELECT(6,2,3,8); SELECT(6,3,2,9); FDIV(10,9,8); \
	FMUL(9,9,10); FADD(8,8,9); FDIV(11,30,8); \
	FMUL(12,0,10); FMUL(13,1,10); FADD(14,0,13); FADD(15,12,1); \
	FSUB(16,1,12); FSUB(17,13,0); SELECT(6,14,15,4); SELECT(6,16,17,5); \
	FMUL(4,4,11); FMUL(5,5,11); \
	VMOV V7.D[0],R4; VMOV V7.D[1],R5; ORR R4,R5,R4; CBZ R4,STORE; \
	FDIV(0,0,31); FDIV(1,1,31); SELECT(7,0,4,14); SELECT(7,1,5,15); \
	VORR V14.B16,V14.B16,V4.B16; VORR V15.B16,V15.B16,V5.B16

TEXT ·complex128NEON(SB),NOSPLIT,$0-73
	MOVD dst_base+0(FP),R0
	MOVD x_base+24(FP),R1
	MOVD y_base+48(FP),R2
	MOVD dst_len+8(FP),R3
	LSR $1,R3,R3
	MOVBU op+72(FP),R4
	CMP $3,R4
	BEQ cdiv
cmulLoop:
	VLD2.P 32(R1),[V0.D2,V1.D2]
	VLD2.P 32(R2),[V2.D2,V3.D2]
	COMPLEX_MUL_WIDE
	VST2.P [V4.D2,V5.D2],32(R0)
	SUBS $1,R3,R3
	BNE cmulLoop
	RET
cdiv:
	FMOVD $1.0,F30
	VDUP V30.D[0],V30.D2
	VEOR V31.B16,V31.B16,V31.B16
cdivLoop:
	VLD2.P 32(R1),[V0.D2,V1.D2]
	VLD2.P 32(R2),[V2.D2,V3.D2]
	COMPLEX_DIV_WIDE(cdivStore)
cdivStore:
	VST2.P [V4.D2,V5.D2],32(R0)
	SUBS $1,R3,R3
	BNE cdivLoop
	RET

// Load two complex64 values, deinterleave, widen in the same registers.
#define CVTL(D,N) WORD $(0x0e617800 | ((N)<<5) | D)
#define CVTN(D,N) WORD $(0x0e616800 | ((N)<<5) | D)
#define LOAD_COMPLEX64 \
	VLD2.P 16(R1),[V0.S2,V1.S2]; VLD2.P 16(R2),[V2.S2,V3.S2]; \
	CVTL(0,0); CVTL(1,1); CVTL(2,2); CVTL(3,3)
#define STORE_COMPLEX64 CVTN(4,4); CVTN(5,5); VST2.P [V4.S2,V5.S2],16(R0)

TEXT ·complex64NEON(SB),NOSPLIT,$0-73
	MOVD dst_base+0(FP),R0
	MOVD x_base+24(FP),R1
	MOVD y_base+48(FP),R2
	MOVD dst_len+8(FP),R3
	LSR $1,R3,R3
	MOVBU op+72(FP),R4
	CMP $3,R4
	BEQ c32div
c32mulLoop:
	LOAD_COMPLEX64
	COMPLEX_MUL_WIDE
	STORE_COMPLEX64
	SUBS $1,R3,R3
	BNE c32mulLoop
	RET
c32div:
	FMOVD $1.0,F30
	VDUP V30.D[0],V30.D2
	VEOR V31.B16,V31.B16,V31.B16
c32divLoop:
	LOAD_COMPLEX64
	COMPLEX_DIV_WIDE(c32divStore)
c32divStore:
	STORE_COMPLEX64
	SUBS $1,R3,R3
	BNE c32divLoop
	RET
