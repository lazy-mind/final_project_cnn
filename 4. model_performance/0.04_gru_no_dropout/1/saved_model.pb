��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*1.15.02v1.15.0-rc3-22-g590d6eef7e8�
�
features/embeddingsVarHandleOp*$
shared_namefeatures/embeddings*
dtype0*
_output_shapes
: *
shape:
��
}
'features/embeddings/Read/ReadVariableOpReadVariableOpfeatures/embeddings*
dtype0* 
_output_shapes
:
��
t
dense/kernelVarHandleOp*
shared_namedense/kernel*
dtype0*
_output_shapes
: *
shape
: 
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes

: 
l

dense/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:
x
training/Adam/iterVarHandleOp*
dtype0	*
_output_shapes
: *
shape: *#
shared_nametraining/Adam/iter
q
&training/Adam/iter/Read/ReadVariableOpReadVariableOptraining/Adam/iter*
dtype0	*
_output_shapes
: 
|
training/Adam/beta_1VarHandleOp*
_output_shapes
: *
shape: *%
shared_nametraining/Adam/beta_1*
dtype0
u
(training/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining/Adam/beta_1*
dtype0*
_output_shapes
: 
|
training/Adam/beta_2VarHandleOp*
shape: *%
shared_nametraining/Adam/beta_2*
dtype0*
_output_shapes
: 
u
(training/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining/Adam/beta_2*
dtype0*
_output_shapes
: 
z
training/Adam/decayVarHandleOp*$
shared_nametraining/Adam/decay*
dtype0*
_output_shapes
: *
shape: 
s
'training/Adam/decay/Read/ReadVariableOpReadVariableOptraining/Adam/decay*
dtype0*
_output_shapes
: 
�
training/Adam/learning_rateVarHandleOp*
dtype0*
_output_shapes
: *
shape: *,
shared_nametraining/Adam/learning_rate
�
/training/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining/Adam/learning_rate*
dtype0*
_output_shapes
: 
p

gru/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape
:`*
shared_name
gru/kernel
i
gru/kernel/Read/ReadVariableOpReadVariableOp
gru/kernel*
dtype0*
_output_shapes

:`
�
gru/recurrent_kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape
: `*%
shared_namegru/recurrent_kernel
}
(gru/recurrent_kernel/Read/ReadVariableOpReadVariableOpgru/recurrent_kernel*
dtype0*
_output_shapes

: `
h
gru/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:`*
shared_name
gru/bias
a
gru/bias/Read/ReadVariableOpReadVariableOpgru/bias*
dtype0*
_output_shapes
:`
^
totalVarHandleOp*
shape: *
shared_nametotal*
dtype0*
_output_shapes
: 
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
b
count_3VarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
dtype0*
_output_shapes
: 
�
#training/Adam/features/embeddings/mVarHandleOp*
_output_shapes
: *
shape:
��*4
shared_name%#training/Adam/features/embeddings/m*
dtype0
�
7training/Adam/features/embeddings/m/Read/ReadVariableOpReadVariableOp#training/Adam/features/embeddings/m*
dtype0* 
_output_shapes
:
��
�
training/Adam/dense/kernel/mVarHandleOp*
shape
: *-
shared_nametraining/Adam/dense/kernel/m*
dtype0*
_output_shapes
: 
�
0training/Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense/kernel/m*
dtype0*
_output_shapes

: 
�
training/Adam/dense/bias/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:*+
shared_nametraining/Adam/dense/bias/m
�
.training/Adam/dense/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense/bias/m*
dtype0*
_output_shapes
:
�
training/Adam/gru/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape
:`*+
shared_nametraining/Adam/gru/kernel/m
�
.training/Adam/gru/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/gru/kernel/m*
dtype0*
_output_shapes

:`
�
$training/Adam/gru/recurrent_kernel/mVarHandleOp*
shape
: `*5
shared_name&$training/Adam/gru/recurrent_kernel/m*
dtype0*
_output_shapes
: 
�
8training/Adam/gru/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp$training/Adam/gru/recurrent_kernel/m*
dtype0*
_output_shapes

: `
�
training/Adam/gru/bias/mVarHandleOp*
_output_shapes
: *
shape:`*)
shared_nametraining/Adam/gru/bias/m*
dtype0
�
,training/Adam/gru/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/gru/bias/m*
dtype0*
_output_shapes
:`
�
#training/Adam/features/embeddings/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:
��*4
shared_name%#training/Adam/features/embeddings/v
�
7training/Adam/features/embeddings/v/Read/ReadVariableOpReadVariableOp#training/Adam/features/embeddings/v*
dtype0* 
_output_shapes
:
��
�
training/Adam/dense/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape
: *-
shared_nametraining/Adam/dense/kernel/v
�
0training/Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense/kernel/v*
dtype0*
_output_shapes

: 
�
training/Adam/dense/bias/vVarHandleOp*
shape:*+
shared_nametraining/Adam/dense/bias/v*
dtype0*
_output_shapes
: 
�
.training/Adam/dense/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense/bias/v*
dtype0*
_output_shapes
:
�
training/Adam/gru/kernel/vVarHandleOp*
shape
:`*+
shared_nametraining/Adam/gru/kernel/v*
dtype0*
_output_shapes
: 
�
.training/Adam/gru/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/gru/kernel/v*
dtype0*
_output_shapes

:`
�
$training/Adam/gru/recurrent_kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape
: `*5
shared_name&$training/Adam/gru/recurrent_kernel/v
�
8training/Adam/gru/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp$training/Adam/gru/recurrent_kernel/v*
_output_shapes

: `*
dtype0
�
training/Adam/gru/bias/vVarHandleOp*
shape:`*)
shared_nametraining/Adam/gru/bias/v*
dtype0*
_output_shapes
: 
�
,training/Adam/gru/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/gru/bias/v*
dtype0*
_output_shapes
:`

NoOpNoOp
�*
ConstConst"/device:CPU:0*
_output_shapes
: *�*
value�*B�* B�*
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
trainable_variables
regularization_losses
	variables
		keras_api


signatures
R
trainable_variables
regularization_losses
	variables
	keras_api
x

embeddings
_callable_losses
trainable_variables
regularization_losses
	variables
	keras_api
�
cell

state_spec
_callable_losses
trainable_variables
regularization_losses
	variables
	keras_api
~

kernel
bias
_callable_losses
trainable_variables
 regularization_losses
!	variables
"	keras_api
�
#iter

$beta_1

%beta_2
	&decay
'learning_ratemUmVmW(mX)mY*mZv[v\v](v^)v_*v`
*
0
(1
)2
*3
4
5
 
*
0
(1
)2
*3
4
5
�
+layer_regularization_losses
trainable_variables
,metrics
regularization_losses
-non_trainable_variables

.layers
	variables
 
 
 
 
�
/layer_regularization_losses
trainable_variables
0metrics
regularization_losses
1non_trainable_variables

2layers
	variables
ca
VARIABLE_VALUEfeatures/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

0
 

0
�
3layer_regularization_losses
trainable_variables
4metrics
regularization_losses
5non_trainable_variables

6layers
	variables
�

(kernel
)recurrent_kernel
*bias
7_callable_losses
8trainable_variables
9regularization_losses
:	variables
;	keras_api
 
 

(0
)1
*2
 

(0
)1
*2
�
<layer_regularization_losses
trainable_variables
=metrics
regularization_losses
>non_trainable_variables

?layers
	variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 

0
1
�
@layer_regularization_losses
trainable_variables
Ametrics
 regularization_losses
Bnon_trainable_variables

Clayers
!	variables
QO
VARIABLE_VALUEtraining/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEtraining/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEtraining/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUE
gru/kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEgru/recurrent_kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEgru/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
 

D0
 

0
1
2
 
 
 
 
 
 
 
 
 

(0
)1
*2
 

(0
)1
*2
�
Elayer_regularization_losses
8trainable_variables
Fmetrics
9regularization_losses
Gnon_trainable_variables

Hlayers
:	variables
 
 
 

0
 
 
 
 
�
	Itotal
	Jcount
K
_fn_kwargs
L_updates
Mtrainable_variables
Nregularization_losses
O	variables
P	keras_api
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_34keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 

I0
J1
�
Qlayer_regularization_losses
Mtrainable_variables
Rmetrics
Nregularization_losses
Snon_trainable_variables

Tlayers
O	variables
 
 

I0
J1
 
��
VARIABLE_VALUE#training/Adam/features/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining/Adam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEtraining/Adam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEtraining/Adam/gru/kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE$training/Adam/gru/recurrent_kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEtraining/Adam/gru/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#training/Adam/features/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining/Adam/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEtraining/Adam/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEtraining/Adam/gru/kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE$training/Adam/gru/recurrent_kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEtraining/Adam/gru/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0
�
serving_default_features_inputPlaceholder*'
_output_shapes
:���������*
shape:���������*
dtype0
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_features_inputfeatures/embeddings
gru/kernelgru/biasgru/recurrent_kerneldense/kernel
dense/bias**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:���������*
Tin
	2*+
_gradient_op_typePartitionedCall-5059*+
f&R$
"__inference_signature_wrapper_3095*
Tout
2
O
saver_filenamePlaceholder*
dtype0*
_output_shapes
: *
shape: 
�

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'features/embeddings/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp&training/Adam/iter/Read/ReadVariableOp(training/Adam/beta_1/Read/ReadVariableOp(training/Adam/beta_2/Read/ReadVariableOp'training/Adam/decay/Read/ReadVariableOp/training/Adam/learning_rate/Read/ReadVariableOpgru/kernel/Read/ReadVariableOp(gru/recurrent_kernel/Read/ReadVariableOpgru/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount_3/Read/ReadVariableOp7training/Adam/features/embeddings/m/Read/ReadVariableOp0training/Adam/dense/kernel/m/Read/ReadVariableOp.training/Adam/dense/bias/m/Read/ReadVariableOp.training/Adam/gru/kernel/m/Read/ReadVariableOp8training/Adam/gru/recurrent_kernel/m/Read/ReadVariableOp,training/Adam/gru/bias/m/Read/ReadVariableOp7training/Adam/features/embeddings/v/Read/ReadVariableOp0training/Adam/dense/kernel/v/Read/ReadVariableOp.training/Adam/dense/bias/v/Read/ReadVariableOp.training/Adam/gru/kernel/v/Read/ReadVariableOp8training/Adam/gru/recurrent_kernel/v/Read/ReadVariableOp,training/Adam/gru/bias/v/Read/ReadVariableOpConst*
Tout
2**
config_proto

GPU 

CPU2J 8*&
Tin
2	*
_output_shapes
: *+
_gradient_op_typePartitionedCall-5106*&
f!R
__inference__traced_save_5105
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamefeatures/embeddingsdense/kernel
dense/biastraining/Adam/itertraining/Adam/beta_1training/Adam/beta_2training/Adam/decaytraining/Adam/learning_rate
gru/kernelgru/recurrent_kernelgru/biastotalcount_3#training/Adam/features/embeddings/mtraining/Adam/dense/kernel/mtraining/Adam/dense/bias/mtraining/Adam/gru/kernel/m$training/Adam/gru/recurrent_kernel/mtraining/Adam/gru/bias/m#training/Adam/features/embeddings/vtraining/Adam/dense/kernel/vtraining/Adam/dense/bias/vtraining/Adam/gru/kernel/v$training/Adam/gru/recurrent_kernel/vtraining/Adam/gru/bias/v*
_output_shapes
: *%
Tin
2*+
_gradient_op_typePartitionedCall-5194*)
f$R"
 __inference__traced_restore_5193*
Tout
2**
config_proto

GPU 

CPU2J 8��
�
�
while_cond_2809
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor

gru_kernel
gru_bias
gru_recurrent_kernel
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*>
_input_shapes-
+: : : : :��������� : : :::: : : : : : : :	 :  : 
�U
�
B__inference_gru_cell_layer_call_and_return_conditional_losses_1934

inputs

states
readvariableop_gru_kernel
readvariableop_3_gru_bias)
%readvariableop_6_gru_recurrent_kernel
identity

identity_1��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�ReadVariableOp_7�ReadVariableOp_8h
ReadVariableOpReadVariableOpreadvariableop_gru_kernel*
dtype0*
_output_shapes

:`d
strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_1Const*
_output_shapes
:*
valueB"        *
dtype0f
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
T0*
Index0b
MatMulMatMulinputsstrided_slice:output:0*
T0*'
_output_shapes
:��������� {
ReadVariableOp_1ReadVariableOpreadvariableop_gru_kernel^ReadVariableOp*
dtype0*
_output_shapes

:`f
strided_slice_1/stackConst*
_output_shapes
:*
valueB"        *
dtype0h
strided_slice_1/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
T0*
Index0f
MatMul_1MatMulinputsstrided_slice_1:output:0*
T0*'
_output_shapes
:��������� }
ReadVariableOp_2ReadVariableOpreadvariableop_gru_kernel^ReadVariableOp_1*
dtype0*
_output_shapes

:`f
strided_slice_2/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_2/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
T0*
Index0f
MatMul_2MatMulinputsstrided_slice_2:output:0*
T0*'
_output_shapes
:��������� f
ReadVariableOp_3ReadVariableOpreadvariableop_3_gru_bias*
dtype0*
_output_shapes
:`_
strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*

begin_mask*
_output_shapes
: *
T0*
Index0p
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*'
_output_shapes
:��������� *
T0y
ReadVariableOp_4ReadVariableOpreadvariableop_3_gru_bias^ReadVariableOp_3*
_output_shapes
:`*
dtype0_
strided_slice_4/stackConst*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_4/stack_1Const*
valueB:@*
dtype0*
_output_shapes
:a
strided_slice_4/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_4StridedSliceReadVariableOp_4:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
_output_shapes
: *
Index0*
T0t
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:��������� y
ReadVariableOp_5ReadVariableOpreadvariableop_3_gru_bias^ReadVariableOp_4*
dtype0*
_output_shapes
:`_
strided_slice_5/stackConst*
dtype0*
_output_shapes
:*
valueB:@a
strided_slice_5/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_5/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_5StridedSliceReadVariableOp_5:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
end_mask*
_output_shapes
: *
Index0*
T0t
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*'
_output_shapes
:��������� *
T0v
ReadVariableOp_6ReadVariableOp%readvariableop_6_gru_recurrent_kernel*
dtype0*
_output_shapes

: `f
strided_slice_6/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_6/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_6/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
strided_slice_6StridedSliceReadVariableOp_6:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:  f
MatMul_3MatMulstatesstrided_slice_6:output:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_7ReadVariableOp%readvariableop_6_gru_recurrent_kernel^ReadVariableOp_6*
dtype0*
_output_shapes

: `f
strided_slice_7/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_7/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_7/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_7StridedSliceReadVariableOp_7:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
Index0*
T0f
MatMul_4MatMulstatesstrided_slice_7:output:0*
T0*'
_output_shapes
:��������� d
addAddV2BiasAdd:output:0MatMul_3:product:0*'
_output_shapes
:��������� *
T0J
ConstConst*
dtype0*
_output_shapes
: *
valueB
 *��L>L
Const_1Const*
valueB
 *   ?*
dtype0*
_output_shapes
: U
MulMuladd:z:0Const:output:0*'
_output_shapes
:��������� *
T0Y
Add_1AddMul:z:0Const_1:output:0*'
_output_shapes
:��������� *
T0\
clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*'
_output_shapes
:��������� *
T0T
clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:��������� h
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:��������� L
Const_2Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_3Const*
dtype0*
_output_shapes
: *
valueB
 *   ?[
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:��������� [
Add_3Add	Mul_1:z:0Const_3:output:0*'
_output_shapes
:��������� *
T0^
clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:��������� V
clip_by_value_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *    �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*'
_output_shapes
:��������� *
T0[
mul_2Mulclip_by_value_1:z:0states*'
_output_shapes
:��������� *
T0�
ReadVariableOp_8ReadVariableOp%readvariableop_6_gru_recurrent_kernel^ReadVariableOp_7*
dtype0*
_output_shapes

: `f
strided_slice_8/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_8/stack_1Const*
dtype0*
_output_shapes
:*
valueB"        h
strided_slice_8/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_slice_8StridedSliceReadVariableOp_8:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
end_mask*
_output_shapes

:  *
Index0*
T0*

begin_maski
MatMul_5MatMul	mul_2:z:0strided_slice_8:output:0*
T0*'
_output_shapes
:��������� h
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*'
_output_shapes
:��������� *
T0I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:��������� Y
mul_3Mulclip_by_value:z:0states*
T0*'
_output_shapes
:��������� J
sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: _
subSubsub/x:output:0clip_by_value:z:0*'
_output_shapes
:��������� *
T0Q
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:��������� �
IdentityIdentity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*'
_output_shapes
:��������� �

Identity_1Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0"
identityIdentity:output:0*E
_input_shapes4
2:���������:��������� :::2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82 
ReadVariableOpReadVariableOp:&"
 
_user_specified_namestates: : : :& "
 
_user_specified_nameinputs
��
�
=__inference_gru_layer_call_and_return_conditional_losses_2958

inputs
readvariableop_gru_kernel
readvariableop_3_gru_bias)
%readvariableop_6_gru_recurrent_kernel
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�ReadVariableOp_7�ReadVariableOp_8�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0M
zeros/mul/yConst*
value	B : *
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
_output_shapes
: *
T0O
zeros/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
_output_shapes
: *
T0P
zeros/packed/1Const*
value	B : *
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*+
_output_shapes
:���������*
T0D
Shape_1Shapetranspose:y:0*
_output_shapes
:*
T0_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0f
TensorArrayV2/element_shapeConst*
valueB :
���������*
dtype0*
_output_shapes
: �
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: �
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
valueB"����   *
dtype0�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:���������*
Index0*
T0h
ReadVariableOpReadVariableOpreadvariableop_gru_kernel*
_output_shapes

:`*
dtype0f
strided_slice_3/stackConst*
dtype0*
_output_shapes
:*
valueB"        h
strided_slice_3/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_3/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_3StridedSliceReadVariableOp:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

: v
MatMulMatMulstrided_slice_2:output:0strided_slice_3:output:0*
T0*'
_output_shapes
:��������� {
ReadVariableOp_1ReadVariableOpreadvariableop_gru_kernel^ReadVariableOp*
dtype0*
_output_shapes

:`f
strided_slice_4/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_4/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_4StridedSliceReadVariableOp_1:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
_output_shapes

: *
T0*
Index0*

begin_mask*
end_maskx
MatMul_1MatMulstrided_slice_2:output:0strided_slice_4:output:0*'
_output_shapes
:��������� *
T0}
ReadVariableOp_2ReadVariableOpreadvariableop_gru_kernel^ReadVariableOp_1*
dtype0*
_output_shapes

:`f
strided_slice_5/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_5/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_5/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
strided_slice_5StridedSliceReadVariableOp_2:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

: x
MatMul_2MatMulstrided_slice_2:output:0strided_slice_5:output:0*'
_output_shapes
:��������� *
T0f
ReadVariableOp_3ReadVariableOpreadvariableop_3_gru_bias*
dtype0*
_output_shapes
:`_
strided_slice_6/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_6/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_6/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_6StridedSliceReadVariableOp_3:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*

begin_mask*
_output_shapes
: *
T0*
Index0p
BiasAddBiasAddMatMul:product:0strided_slice_6:output:0*'
_output_shapes
:��������� *
T0y
ReadVariableOp_4ReadVariableOpreadvariableop_3_gru_bias^ReadVariableOp_3*
dtype0*
_output_shapes
:`_
strided_slice_7/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_7/stack_1Const*
valueB:@*
dtype0*
_output_shapes
:a
strided_slice_7/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
strided_slice_7StridedSliceReadVariableOp_4:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: t
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_7:output:0*'
_output_shapes
:��������� *
T0y
ReadVariableOp_5ReadVariableOpreadvariableop_3_gru_bias^ReadVariableOp_4*
dtype0*
_output_shapes
:`_
strided_slice_8/stackConst*
valueB:@*
dtype0*
_output_shapes
:a
strided_slice_8/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_8/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_8StridedSliceReadVariableOp_5:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
end_mask*
_output_shapes
: *
Index0*
T0t
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_8:output:0*
T0*'
_output_shapes
:��������� v
ReadVariableOp_6ReadVariableOp%readvariableop_6_gru_recurrent_kernel*
dtype0*
_output_shapes

: `f
strided_slice_9/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_9/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_9/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_9StridedSliceReadVariableOp_6:value:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
T0*
Index0n
MatMul_3MatMulzeros:output:0strided_slice_9:output:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_7ReadVariableOp%readvariableop_6_gru_recurrent_kernel^ReadVariableOp_6*
dtype0*
_output_shapes

: `g
strided_slice_10/stackConst*
dtype0*
_output_shapes
:*
valueB"        i
strided_slice_10/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:i
strided_slice_10/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_10StridedSliceReadVariableOp_7:value:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:  o
MatMul_4MatMulzeros:output:0strided_slice_10:output:0*'
_output_shapes
:��������� *
T0d
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:��������� J
ConstConst*
_output_shapes
: *
valueB
 *��L>*
dtype0L
Const_1Const*
valueB
 *   ?*
dtype0*
_output_shapes
: U
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:��������� Y
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:��������� \
clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:��������� T
clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:��������� h
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*'
_output_shapes
:��������� *
T0L
Const_2Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_3Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_2:output:0*'
_output_shapes
:��������� *
T0[
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:��������� ^
clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:��������� V
clip_by_value_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*'
_output_shapes
:��������� *
T0c
mul_2Mulclip_by_value_1:z:0zeros:output:0*'
_output_shapes
:��������� *
T0�
ReadVariableOp_8ReadVariableOp%readvariableop_6_gru_recurrent_kernel^ReadVariableOp_7*
dtype0*
_output_shapes

: `g
strided_slice_11/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:i
strided_slice_11/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:i
strided_slice_11/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_11StridedSliceReadVariableOp_8:value:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
T0*
Index0j
MatMul_5MatMul	mul_2:z:0strided_slice_11:output:0*
T0*'
_output_shapes
:��������� h
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:��������� a
mul_3Mulclip_by_value:z:0zeros:output:0*
T0*'
_output_shapes
:��������� J
sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: _
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:��������� Q
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
valueB"����    *
dtype0*
_output_shapes
:�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
dtype0*
_output_shapes
: *
value	B : c
while/maximum_iterationsConst*
dtype0*
_output_shapes
: *
valueB :
���������T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0readvariableop_gru_kernelreadvariableop_3_gru_bias%readvariableop_6_gru_recurrent_kernel^ReadVariableOp_2^ReadVariableOp_5^ReadVariableOp_8*
parallel_iterations *
condR
while_cond_2809*
_num_original_outputs
*
bodyR
while_body_2810*9
_output_shapes'
%: : : : :��������� : : : : : *
T
2
*8
output_shapes'
%: : : : :��������� : : : : : *
_lower_using_switch_merge(K
while/IdentityIdentitywhile:output:0*
_output_shapes
: *
T0M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
_output_shapes
: *
T0^
while/Identity_4Identitywhile:output:4*'
_output_shapes
:��������� *
T0M
while/Identity_5Identitywhile:output:5*
T0*
_output_shapes
: M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
_output_shapes
: *
T0M
while/Identity_8Identitywhile:output:8*
_output_shapes
: *
T0M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"����    *
dtype0*
_output_shapes
:�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:��������� i
strided_slice_12/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:b
strided_slice_12/stack_1Const*
valueB: *
dtype0*
_output_shapes
:b
strided_slice_12/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
strided_slice_12StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:��������� *
T0*
Index0e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� �
IdentityIdentitystrided_slice_12:output:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8^while*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*6
_input_shapes%
#:���������:::2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82
whilewhile2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs: : : 
�U
�
B__inference_gru_cell_layer_call_and_return_conditional_losses_4983

inputs
states_0
readvariableop_gru_kernel
readvariableop_3_gru_bias)
%readvariableop_6_gru_recurrent_kernel
identity

identity_1��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�ReadVariableOp_7�ReadVariableOp_8h
ReadVariableOpReadVariableOpreadvariableop_gru_kernel*
dtype0*
_output_shapes

:`d
strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
T0*
Index0b
MatMulMatMulinputsstrided_slice:output:0*'
_output_shapes
:��������� *
T0{
ReadVariableOp_1ReadVariableOpreadvariableop_gru_kernel^ReadVariableOp*
dtype0*
_output_shapes

:`f
strided_slice_1/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB"    @   h
strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

: f
MatMul_1MatMulinputsstrided_slice_1:output:0*
T0*'
_output_shapes
:��������� }
ReadVariableOp_2ReadVariableOpreadvariableop_gru_kernel^ReadVariableOp_1*
dtype0*
_output_shapes

:`f
strided_slice_2/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
Index0*
T0f
MatMul_2MatMulinputsstrided_slice_2:output:0*
T0*'
_output_shapes
:��������� f
ReadVariableOp_3ReadVariableOpreadvariableop_3_gru_bias*
dtype0*
_output_shapes
:`_
strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*

begin_mask*
_output_shapes
: *
T0*
Index0p
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*'
_output_shapes
:��������� y
ReadVariableOp_4ReadVariableOpreadvariableop_3_gru_bias^ReadVariableOp_3*
dtype0*
_output_shapes
:`_
strided_slice_4/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_4/stack_1Const*
valueB:@*
dtype0*
_output_shapes
:a
strided_slice_4/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_4StridedSliceReadVariableOp_4:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
T0*
Index0*
_output_shapes
: t
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:��������� y
ReadVariableOp_5ReadVariableOpreadvariableop_3_gru_bias^ReadVariableOp_4*
dtype0*
_output_shapes
:`_
strided_slice_5/stackConst*
valueB:@*
dtype0*
_output_shapes
:a
strided_slice_5/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_5/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_5StridedSliceReadVariableOp_5:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
T0*
Index0*
end_mask*
_output_shapes
: t
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:��������� v
ReadVariableOp_6ReadVariableOp%readvariableop_6_gru_recurrent_kernel*
dtype0*
_output_shapes

: `f
strided_slice_6/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_6/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_6/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_slice_6StridedSliceReadVariableOp_6:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:  h
MatMul_3MatMulstates_0strided_slice_6:output:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_7ReadVariableOp%readvariableop_6_gru_recurrent_kernel^ReadVariableOp_6*
dtype0*
_output_shapes

: `f
strided_slice_7/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_7/stack_1Const*
_output_shapes
:*
valueB"    @   *
dtype0h
strided_slice_7/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_7StridedSliceReadVariableOp_7:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:  h
MatMul_4MatMulstates_0strided_slice_7:output:0*
T0*'
_output_shapes
:��������� d
addAddV2BiasAdd:output:0MatMul_3:product:0*'
_output_shapes
:��������� *
T0J
ConstConst*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_1Const*
valueB
 *   ?*
dtype0*
_output_shapes
: U
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:��������� Y
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:��������� \
clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:��������� T
clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:��������� h
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:��������� L
Const_2Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_3Const*
_output_shapes
: *
valueB
 *   ?*
dtype0[
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:��������� [
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:��������� ^
clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*'
_output_shapes
:��������� *
T0V
clip_by_value_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*'
_output_shapes
:��������� *
T0]
mul_2Mulclip_by_value_1:z:0states_0*'
_output_shapes
:��������� *
T0�
ReadVariableOp_8ReadVariableOp%readvariableop_6_gru_recurrent_kernel^ReadVariableOp_7*
_output_shapes

: `*
dtype0f
strided_slice_8/stackConst*
_output_shapes
:*
valueB"    @   *
dtype0h
strided_slice_8/stack_1Const*
_output_shapes
:*
valueB"        *
dtype0h
strided_slice_8/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_8StridedSliceReadVariableOp_8:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
T0*
Index0i
MatMul_5MatMul	mul_2:z:0strided_slice_8:output:0*'
_output_shapes
:��������� *
T0h
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*'
_output_shapes
:��������� *
T0I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:��������� [
mul_3Mulclip_by_value:z:0states_0*
T0*'
_output_shapes
:��������� J
sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: _
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:��������� Q
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_5AddV2	mul_3:z:0	mul_4:z:0*'
_output_shapes
:��������� *
T0�
IdentityIdentity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*'
_output_shapes
:��������� �

Identity_1Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:���������:��������� :::2$
ReadVariableOp_8ReadVariableOp_82 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_7: : :& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0: 
�
�
"__inference_gru_layer_call_fn_4239

inputs&
"statefulpartitionedcall_gru_kernel$
 statefulpartitionedcall_gru_bias0
,statefulpartitionedcall_gru_recurrent_kernel
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs"statefulpartitionedcall_gru_kernel statefulpartitionedcall_gru_bias,statefulpartitionedcall_gru_recurrent_kernel*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:��������� *
Tin
2*+
_gradient_op_typePartitionedCall-2970*F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_2958�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*6
_input_shapes%
#:���������:::22
StatefulPartitionedCallStatefulPartitionedCall: : :& "
 
_user_specified_nameinputs: 
�
�
while_body_2198
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0(
$statefulpartitionedcall_gru_kernel_0&
"statefulpartitionedcall_gru_bias_02
.statefulpartitionedcall_gru_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor&
"statefulpartitionedcall_gru_kernel$
 statefulpartitionedcall_gru_bias0
,statefulpartitionedcall_gru_recurrent_kernel��StatefulPartitionedCall�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:����������
StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2$statefulpartitionedcall_gru_kernel_0"statefulpartitionedcall_gru_bias_0.statefulpartitionedcall_gru_recurrent_kernel_0**
config_proto

GPU 

CPU2J 8*
Tin	
2*:
_output_shapes(
&:��������� :��������� *+
_gradient_op_typePartitionedCall-1939*K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_1842*
Tout
2�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder StatefulPartitionedCall:output:0*
element_dtype0*
_output_shapes
: G
add/yConst*
value	B :*
dtype0*
_output_shapes
: J
addAddV2placeholderadd/y:output:0*
_output_shapes
: *
T0I
add_1/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: Z
IdentityIdentity	add_1:z:0^StatefulPartitionedCall*
T0*
_output_shapes
: k

Identity_1Identitywhile_maximum_iterations^StatefulPartitionedCall*
T0*
_output_shapes
: Z

Identity_2Identityadd:z:0^StatefulPartitionedCall*
T0*
_output_shapes
: �

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^StatefulPartitionedCall*
T0*
_output_shapes
: �

Identity_4Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� "$
strided_slice_1strided_slice_1_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"J
"statefulpartitionedcall_gru_kernel$statefulpartitionedcall_gru_kernel_0"!

identity_4Identity_4:output:0"^
,statefulpartitionedcall_gru_recurrent_kernel.statefulpartitionedcall_gru_recurrent_kernel_0"
identityIdentity:output:0"F
 statefulpartitionedcall_gru_bias"statefulpartitionedcall_gru_bias_0*>
_input_shapes-
+: : : : :��������� : : :::22
StatefulPartitionedCallStatefulPartitionedCall:  : : : : : : : : :	 
�
�
while_cond_4356
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor

gru_kernel
gru_bias
gru_recurrent_kernel
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*>
_input_shapes-
+: : : : :��������� : : ::::  : : : : : : : : :	 
��
�
=__inference_gru_layer_call_and_return_conditional_losses_2692

inputs
readvariableop_gru_kernel
readvariableop_3_gru_bias)
%readvariableop_6_gru_recurrent_kernel
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�ReadVariableOp_7�ReadVariableOp_8�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0_
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B : *
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
dtype0*
_output_shapes
: *
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0f
TensorArrayV2/element_shapeConst*
valueB :
���������*
dtype0*
_output_shapes
: �
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: �
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
element_dtype0*
_output_shapes
: *

shape_type0_
strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*'
_output_shapes
:���������*
Index0*
T0*
shrink_axis_maskh
ReadVariableOpReadVariableOpreadvariableop_gru_kernel*
_output_shapes

:`*
dtype0f
strided_slice_3/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_3/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_3/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_3StridedSliceReadVariableOp:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

: v
MatMulMatMulstrided_slice_2:output:0strided_slice_3:output:0*
T0*'
_output_shapes
:��������� {
ReadVariableOp_1ReadVariableOpreadvariableop_gru_kernel^ReadVariableOp*
dtype0*
_output_shapes

:`f
strided_slice_4/stackConst*
_output_shapes
:*
valueB"        *
dtype0h
strided_slice_4/stack_1Const*
_output_shapes
:*
valueB"    @   *
dtype0h
strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_4StridedSliceReadVariableOp_1:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
T0*
Index0x
MatMul_1MatMulstrided_slice_2:output:0strided_slice_4:output:0*'
_output_shapes
:��������� *
T0}
ReadVariableOp_2ReadVariableOpreadvariableop_gru_kernel^ReadVariableOp_1*
dtype0*
_output_shapes

:`f
strided_slice_5/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_5/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_5/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_5StridedSliceReadVariableOp_2:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

: x
MatMul_2MatMulstrided_slice_2:output:0strided_slice_5:output:0*
T0*'
_output_shapes
:��������� f
ReadVariableOp_3ReadVariableOpreadvariableop_3_gru_bias*
dtype0*
_output_shapes
:`_
strided_slice_6/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_6/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_6/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_6StridedSliceReadVariableOp_3:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*

begin_mask*
_output_shapes
: p
BiasAddBiasAddMatMul:product:0strided_slice_6:output:0*'
_output_shapes
:��������� *
T0y
ReadVariableOp_4ReadVariableOpreadvariableop_3_gru_bias^ReadVariableOp_3*
dtype0*
_output_shapes
:`_
strided_slice_7/stackConst*
_output_shapes
:*
valueB: *
dtype0a
strided_slice_7/stack_1Const*
valueB:@*
dtype0*
_output_shapes
:a
strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_7StridedSliceReadVariableOp_4:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: t
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_7:output:0*
T0*'
_output_shapes
:��������� y
ReadVariableOp_5ReadVariableOpreadvariableop_3_gru_bias^ReadVariableOp_4*
dtype0*
_output_shapes
:`_
strided_slice_8/stackConst*
valueB:@*
dtype0*
_output_shapes
:a
strided_slice_8/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_8/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_8StridedSliceReadVariableOp_5:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
end_mask*
_output_shapes
: *
Index0*
T0t
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_8:output:0*
T0*'
_output_shapes
:��������� v
ReadVariableOp_6ReadVariableOp%readvariableop_6_gru_recurrent_kernel*
dtype0*
_output_shapes

: `f
strided_slice_9/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_9/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_9/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_9StridedSliceReadVariableOp_6:value:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
Index0*
T0n
MatMul_3MatMulzeros:output:0strided_slice_9:output:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_7ReadVariableOp%readvariableop_6_gru_recurrent_kernel^ReadVariableOp_6*
dtype0*
_output_shapes

: `g
strided_slice_10/stackConst*
valueB"        *
dtype0*
_output_shapes
:i
strided_slice_10/stack_1Const*
dtype0*
_output_shapes
:*
valueB"    @   i
strided_slice_10/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_10StridedSliceReadVariableOp_7:value:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
Index0*
T0o
MatMul_4MatMulzeros:output:0strided_slice_10:output:0*'
_output_shapes
:��������� *
T0d
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:��������� J
ConstConst*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_1Const*
_output_shapes
: *
valueB
 *   ?*
dtype0U
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:��������� Y
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:��������� \
clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:��������� T
clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:��������� h
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:��������� L
Const_2Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_3Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_2:output:0*'
_output_shapes
:��������� *
T0[
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:��������� ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
valueB
 *  �?*
dtype0�
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:��������� V
clip_by_value_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*'
_output_shapes
:��������� *
T0c
mul_2Mulclip_by_value_1:z:0zeros:output:0*'
_output_shapes
:��������� *
T0�
ReadVariableOp_8ReadVariableOp%readvariableop_6_gru_recurrent_kernel^ReadVariableOp_7*
dtype0*
_output_shapes

: `g
strided_slice_11/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:i
strided_slice_11/stack_1Const*
dtype0*
_output_shapes
:*
valueB"        i
strided_slice_11/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_slice_11StridedSliceReadVariableOp_8:value:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:  j
MatMul_5MatMul	mul_2:z:0strided_slice_11:output:0*
T0*'
_output_shapes
:��������� h
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:��������� a
mul_3Mulclip_by_value:z:0zeros:output:0*'
_output_shapes
:��������� *
T0J
sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: _
subSubsub/x:output:0clip_by_value:z:0*'
_output_shapes
:��������� *
T0Q
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_5AddV2	mul_3:z:0	mul_4:z:0*'
_output_shapes
:��������� *
T0n
TensorArrayV2_1/element_shapeConst*
valueB"����    *
dtype0*
_output_shapes
:�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
valueB :
���������*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0readvariableop_gru_kernelreadvariableop_3_gru_bias%readvariableop_6_gru_recurrent_kernel^ReadVariableOp_2^ReadVariableOp_5^ReadVariableOp_8*
condR
while_cond_2543*
_num_original_outputs
*
bodyR
while_body_2544*9
_output_shapes'
%: : : : :��������� : : : : : *
T
2
*8
output_shapes'
%: : : : :��������� : : : : : *
_lower_using_switch_merge(*
parallel_iterations K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
_output_shapes
: *
T0M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:��������� M
while/Identity_5Identitywhile:output:5*
T0*
_output_shapes
: M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
_output_shapes
: *
T0M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
valueB"����    *
dtype0�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:��������� i
strided_slice_12/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:b
strided_slice_12/stack_1Const*
valueB: *
dtype0*
_output_shapes
:b
strided_slice_12/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
strided_slice_12StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*'
_output_shapes
:��������� *
T0*
Index0*
shrink_axis_maske
transpose_1/permConst*
dtype0*
_output_shapes
:*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� �
IdentityIdentitystrided_slice_12:output:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8^while*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*6
_input_shapes%
#:���������:::2$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82
whilewhile2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1: : : :& "
 
_user_specified_nameinputs
�	
�
'__inference_gru_cell_layer_call_fn_5005

inputs
states_0&
"statefulpartitionedcall_gru_kernel$
 statefulpartitionedcall_gru_bias0
,statefulpartitionedcall_gru_recurrent_kernel
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0"statefulpartitionedcall_gru_kernel statefulpartitionedcall_gru_bias,statefulpartitionedcall_gru_recurrent_kernel*+
_gradient_op_typePartitionedCall-1954*K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_1934*
Tout
2**
config_proto

GPU 

CPU2J 8*:
_output_shapes(
&:��������� :��������� *
Tin	
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� �

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*'
_output_shapes
:��������� *
T0"!

identity_1Identity_1:output:0"
identityIdentity:output:0*E
_input_shapes4
2:���������:��������� :::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0: : : 
�
�
gru_while_cond_3498
gru_while_loop_counter 
gru_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_gru_strided_slice_1/
+gru_tensorarrayunstack_tensorlistfromtensor

gru_kernel
gru_bias
gru_recurrent_kernel
identity
T
LessLessplaceholderless_gru_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*>
_input_shapes-
+: : : : :��������� : : :::: : : : : : : :	 :  : 
�	
�
"__inference_signature_wrapper_3095
features_input/
+statefulpartitionedcall_features_embeddings&
"statefulpartitionedcall_gru_kernel$
 statefulpartitionedcall_gru_bias0
,statefulpartitionedcall_gru_recurrent_kernel(
$statefulpartitionedcall_dense_kernel&
"statefulpartitionedcall_dense_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallfeatures_input+statefulpartitionedcall_features_embeddings"statefulpartitionedcall_gru_kernel statefulpartitionedcall_gru_bias,statefulpartitionedcall_gru_recurrent_kernel$statefulpartitionedcall_dense_kernel"statefulpartitionedcall_dense_bias*+
_gradient_op_typePartitionedCall-3086*(
f#R!
__inference__wrapped_model_1718*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:���������*
Tin
	2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:���������*
T0"
identityIdentity:output:0*>
_input_shapes-
+:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : :. *
(
_user_specified_namefeatures_input
�
�
while_cond_2543
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor

gru_kernel
gru_bias
gru_recurrent_kernel
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*>
_input_shapes-
+: : : : :��������� : : ::::	 :  : : : : : : : : 
�j
�
while_body_4357
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_gru_kernel_0
readvariableop_3_gru_bias_0+
'readvariableop_6_gru_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
readvariableop_gru_kernel
readvariableop_3_gru_bias)
%readvariableop_6_gru_recurrent_kernel��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�ReadVariableOp_7�ReadVariableOp_8�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:���������j
ReadVariableOpReadVariableOpreadvariableop_gru_kernel_0*
dtype0*
_output_shapes

:`d
strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
Index0*
T0�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice:output:0*
T0*'
_output_shapes
:��������� }
ReadVariableOp_1ReadVariableOpreadvariableop_gru_kernel_0^ReadVariableOp*
dtype0*
_output_shapes

:`f
strided_slice_2/stackConst*
_output_shapes
:*
valueB"        *
dtype0h
strided_slice_2/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
strided_slice_2StridedSliceReadVariableOp_1:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

: �
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_2:output:0*
T0*'
_output_shapes
:��������� 
ReadVariableOp_2ReadVariableOpreadvariableop_gru_kernel_0^ReadVariableOp_1*
dtype0*
_output_shapes

:`f
strided_slice_3/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB"        h
strided_slice_3/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_3StridedSliceReadVariableOp_2:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

: �
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_3:output:0*'
_output_shapes
:��������� *
T0h
ReadVariableOp_3ReadVariableOpreadvariableop_3_gru_bias_0*
dtype0*
_output_shapes
:`_
strided_slice_4/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_4/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_4/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_4StridedSliceReadVariableOp_3:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*

begin_mask*
_output_shapes
: p
BiasAddBiasAddMatMul:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:��������� {
ReadVariableOp_4ReadVariableOpreadvariableop_3_gru_bias_0^ReadVariableOp_3*
dtype0*
_output_shapes
:`_
strided_slice_5/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_5/stack_1Const*
valueB:@*
dtype0*
_output_shapes
:a
strided_slice_5/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_5StridedSliceReadVariableOp_4:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
_output_shapes
: *
T0*
Index0t
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:��������� {
ReadVariableOp_5ReadVariableOpreadvariableop_3_gru_bias_0^ReadVariableOp_4*
dtype0*
_output_shapes
:`_
strided_slice_6/stackConst*
valueB:@*
dtype0*
_output_shapes
:a
strided_slice_6/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_6/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_6StridedSliceReadVariableOp_5:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
end_mask*
_output_shapes
: *
Index0*
T0t
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_6:output:0*
T0*'
_output_shapes
:��������� x
ReadVariableOp_6ReadVariableOp'readvariableop_6_gru_recurrent_kernel_0*
dtype0*
_output_shapes

: `f
strided_slice_7/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_7/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_7/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_7StridedSliceReadVariableOp_6:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
T0*
Index0m
MatMul_3MatMulplaceholder_2strided_slice_7:output:0*'
_output_shapes
:��������� *
T0�
ReadVariableOp_7ReadVariableOp'readvariableop_6_gru_recurrent_kernel_0^ReadVariableOp_6*
dtype0*
_output_shapes

: `f
strided_slice_8/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_8/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_8/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_8StridedSliceReadVariableOp_7:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:  m
MatMul_4MatMulplaceholder_2strided_slice_8:output:0*
T0*'
_output_shapes
:��������� d
addAddV2BiasAdd:output:0MatMul_3:product:0*'
_output_shapes
:��������� *
T0J
ConstConst*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_1Const*
valueB
 *   ?*
dtype0*
_output_shapes
: U
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:��������� Y
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:��������� \
clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:��������� T
clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:��������� h
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:��������� L
Const_2Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_3Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:��������� [
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:��������� ^
clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*'
_output_shapes
:��������� *
T0V
clip_by_value_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:��������� b
mul_2Mulclip_by_value_1:z:0placeholder_2*'
_output_shapes
:��������� *
T0�
ReadVariableOp_8ReadVariableOp'readvariableop_6_gru_recurrent_kernel_0^ReadVariableOp_7*
dtype0*
_output_shapes

: `f
strided_slice_9/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_9/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_9/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_9StridedSliceReadVariableOp_8:value:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
T0*
Index0i
MatMul_5MatMul	mul_2:z:0strided_slice_9:output:0*
T0*'
_output_shapes
:��������� h
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:��������� `
mul_3Mulclip_by_value:z:0placeholder_2*'
_output_shapes
:��������� *
T0J
sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: _
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:��������� Q
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:��������� �
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_5:z:0*
element_dtype0*
_output_shapes
: I
add_6/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_6AddV2placeholderadd_6/y:output:0*
T0*
_output_shapes
: I
add_7/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_7AddV2while_loop_counteradd_7/y:output:0*
_output_shapes
: *
T0�
IdentityIdentity	add_7:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: �

Identity_1Identitywhile_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
_output_shapes
: *
T0�

Identity_2Identity	add_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
_output_shapes
: *
T0�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: �

Identity_4Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0"$
strided_slice_1strided_slice_1_0"P
%readvariableop_6_gru_recurrent_kernel'readvariableop_6_gru_recurrent_kernel_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"8
readvariableop_3_gru_biasreadvariableop_3_gru_bias_0"8
readvariableop_gru_kernelreadvariableop_gru_kernel_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*>
_input_shapes-
+: : : : :��������� : : :::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_8:  : : : : : : : : :	 
�
�
D__inference_sequential_layer_call_and_return_conditional_losses_3012
features_input8
4features_statefulpartitionedcall_features_embeddings*
&gru_statefulpartitionedcall_gru_kernel(
$gru_statefulpartitionedcall_gru_bias4
0gru_statefulpartitionedcall_gru_recurrent_kernel.
*dense_statefulpartitionedcall_dense_kernel,
(dense_statefulpartitionedcall_dense_bias
identity��dense/StatefulPartitionedCall� features/StatefulPartitionedCall�gru/StatefulPartitionedCall�
 features/StatefulPartitionedCallStatefulPartitionedCallfeatures_input4features_statefulpartitionedcall_features_embeddings**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:���������*+
_gradient_op_typePartitionedCall-2417*K
fFRD
B__inference_features_layer_call_and_return_conditional_losses_2410*
Tout
2�
gru/StatefulPartitionedCallStatefulPartitionedCall)features/StatefulPartitionedCall:output:0&gru_statefulpartitionedcall_gru_kernel$gru_statefulpartitionedcall_gru_bias0gru_statefulpartitionedcall_gru_recurrent_kernel**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:��������� *+
_gradient_op_typePartitionedCall-2961*F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_2692*
Tout
2�
dense/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0*dense_statefulpartitionedcall_dense_kernel(dense_statefulpartitionedcall_dense_bias*
Tin
2*'
_output_shapes
:���������*+
_gradient_op_typePartitionedCall-2999*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_2992*
Tout
2**
config_proto

GPU 

CPU2J 8�
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall!^features/StatefulPartitionedCall^gru/StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*>
_input_shapes-
+:���������::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2D
 features/StatefulPartitionedCall features/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:. *
(
_user_specified_namefeatures_input: : : : : : 
��
�
=__inference_gru_layer_call_and_return_conditional_losses_4223

inputs
readvariableop_gru_kernel
readvariableop_3_gru_bias)
%readvariableop_6_gru_recurrent_kernel
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�ReadVariableOp_7�ReadVariableOp_8�while;
ShapeShapeinputs*
_output_shapes
:*
T0]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0M
zeros/mul/yConst*
value	B : *
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
_output_shapes
: *
T0O
zeros/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
_output_shapes
: *
T0P
zeros/packed/1Const*
value	B : *
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*'
_output_shapes
:��������� *
T0c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
dtype0*
_output_shapes
: *
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: �
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*'
_output_shapes
:���������h
ReadVariableOpReadVariableOpreadvariableop_gru_kernel*
dtype0*
_output_shapes

:`f
strided_slice_3/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_3/stack_1Const*
_output_shapes
:*
valueB"        *
dtype0h
strided_slice_3/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_slice_3StridedSliceReadVariableOp:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
T0*
Index0v
MatMulMatMulstrided_slice_2:output:0strided_slice_3:output:0*
T0*'
_output_shapes
:��������� {
ReadVariableOp_1ReadVariableOpreadvariableop_gru_kernel^ReadVariableOp*
dtype0*
_output_shapes

:`f
strided_slice_4/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_4/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_4StridedSliceReadVariableOp_1:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
Index0*
T0x
MatMul_1MatMulstrided_slice_2:output:0strided_slice_4:output:0*
T0*'
_output_shapes
:��������� }
ReadVariableOp_2ReadVariableOpreadvariableop_gru_kernel^ReadVariableOp_1*
dtype0*
_output_shapes

:`f
strided_slice_5/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_5/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_5/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_5StridedSliceReadVariableOp_2:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

: x
MatMul_2MatMulstrided_slice_2:output:0strided_slice_5:output:0*'
_output_shapes
:��������� *
T0f
ReadVariableOp_3ReadVariableOpreadvariableop_3_gru_bias*
dtype0*
_output_shapes
:`_
strided_slice_6/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_6/stack_1Const*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_6/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_6StridedSliceReadVariableOp_3:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
T0*
Index0*

begin_mask*
_output_shapes
: p
BiasAddBiasAddMatMul:product:0strided_slice_6:output:0*'
_output_shapes
:��������� *
T0y
ReadVariableOp_4ReadVariableOpreadvariableop_3_gru_bias^ReadVariableOp_3*
_output_shapes
:`*
dtype0_
strided_slice_7/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_7/stack_1Const*
valueB:@*
dtype0*
_output_shapes
:a
strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_7StridedSliceReadVariableOp_4:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
_output_shapes
: *
Index0*
T0t
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_7:output:0*'
_output_shapes
:��������� *
T0y
ReadVariableOp_5ReadVariableOpreadvariableop_3_gru_bias^ReadVariableOp_4*
dtype0*
_output_shapes
:`_
strided_slice_8/stackConst*
valueB:@*
dtype0*
_output_shapes
:a
strided_slice_8/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_8/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_8StridedSliceReadVariableOp_5:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
end_mask*
_output_shapes
: *
Index0*
T0t
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_8:output:0*
T0*'
_output_shapes
:��������� v
ReadVariableOp_6ReadVariableOp%readvariableop_6_gru_recurrent_kernel*
dtype0*
_output_shapes

: `f
strided_slice_9/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_9/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_9/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_slice_9StridedSliceReadVariableOp_6:value:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:  n
MatMul_3MatMulzeros:output:0strided_slice_9:output:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_7ReadVariableOp%readvariableop_6_gru_recurrent_kernel^ReadVariableOp_6*
dtype0*
_output_shapes

: `g
strided_slice_10/stackConst*
valueB"        *
dtype0*
_output_shapes
:i
strided_slice_10/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:i
strided_slice_10/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_slice_10StridedSliceReadVariableOp_7:value:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:  o
MatMul_4MatMulzeros:output:0strided_slice_10:output:0*'
_output_shapes
:��������� *
T0d
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:��������� J
ConstConst*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_1Const*
valueB
 *   ?*
dtype0*
_output_shapes
: U
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:��������� Y
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:��������� \
clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*'
_output_shapes
:��������� *
T0T
clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*'
_output_shapes
:��������� *
T0h
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:��������� L
Const_2Const*
_output_shapes
: *
valueB
 *��L>*
dtype0L
Const_3Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:��������� [
Add_3Add	Mul_1:z:0Const_3:output:0*'
_output_shapes
:��������� *
T0^
clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:��������� V
clip_by_value_1/yConst*
_output_shapes
: *
valueB
 *    *
dtype0�
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:��������� c
mul_2Mulclip_by_value_1:z:0zeros:output:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_8ReadVariableOp%readvariableop_6_gru_recurrent_kernel^ReadVariableOp_7*
dtype0*
_output_shapes

: `g
strided_slice_11/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:i
strided_slice_11/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:i
strided_slice_11/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_11StridedSliceReadVariableOp_8:value:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
T0*
Index0j
MatMul_5MatMul	mul_2:z:0strided_slice_11:output:0*
T0*'
_output_shapes
:��������� h
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_4:z:0*'
_output_shapes
:��������� *
T0a
mul_3Mulclip_by_value:z:0zeros:output:0*'
_output_shapes
:��������� *
T0J
sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: _
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:��������� Q
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"����    �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
valueB :
���������*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0readvariableop_gru_kernelreadvariableop_3_gru_bias%readvariableop_6_gru_recurrent_kernel^ReadVariableOp_2^ReadVariableOp_5^ReadVariableOp_8*
T
2
*8
output_shapes'
%: : : : :��������� : : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_4074*
_num_original_outputs
*
bodyR
while_body_4075*9
_output_shapes'
%: : : : :��������� : : : : : K
while/IdentityIdentitywhile:output:0*
_output_shapes
: *
T0M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*'
_output_shapes
:��������� *
T0M
while/Identity_5Identitywhile:output:5*
T0*
_output_shapes
: M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
_output_shapes
: *
T0M
while/Identity_8Identitywhile:output:8*
_output_shapes
: *
T0M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"����    *
dtype0*
_output_shapes
:�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:��������� i
strided_slice_12/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:b
strided_slice_12/stack_1Const*
valueB: *
dtype0*
_output_shapes
:b
strided_slice_12/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
strided_slice_12StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:��������� *
T0*
Index0e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� �
IdentityIdentitystrided_slice_12:output:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8^while*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*6
_input_shapes%
#:���������:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82
whilewhile:& "
 
_user_specified_nameinputs: : : 
�8
�
__inference__traced_save_5105
file_prefix2
.savev2_features_embeddings_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop1
-savev2_training_adam_iter_read_readvariableop	3
/savev2_training_adam_beta_1_read_readvariableop3
/savev2_training_adam_beta_2_read_readvariableop2
.savev2_training_adam_decay_read_readvariableop:
6savev2_training_adam_learning_rate_read_readvariableop)
%savev2_gru_kernel_read_readvariableop3
/savev2_gru_recurrent_kernel_read_readvariableop'
#savev2_gru_bias_read_readvariableop$
 savev2_total_read_readvariableop&
"savev2_count_3_read_readvariableopB
>savev2_training_adam_features_embeddings_m_read_readvariableop;
7savev2_training_adam_dense_kernel_m_read_readvariableop9
5savev2_training_adam_dense_bias_m_read_readvariableop9
5savev2_training_adam_gru_kernel_m_read_readvariableopC
?savev2_training_adam_gru_recurrent_kernel_m_read_readvariableop7
3savev2_training_adam_gru_bias_m_read_readvariableopB
>savev2_training_adam_features_embeddings_v_read_readvariableop;
7savev2_training_adam_dense_kernel_v_read_readvariableop9
5savev2_training_adam_dense_bias_v_read_readvariableop9
5savev2_training_adam_gru_kernel_v_read_readvariableopC
?savev2_training_adam_gru_recurrent_kernel_v_read_readvariableop7
3savev2_training_adam_gru_bias_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_2c2371d8f72148d4a9a19e2547dd5441/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*�
value�B�B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:�
SaveV2/shape_and_slicesConst"/device:CPU:0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_features_embeddings_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop-savev2_training_adam_iter_read_readvariableop/savev2_training_adam_beta_1_read_readvariableop/savev2_training_adam_beta_2_read_readvariableop.savev2_training_adam_decay_read_readvariableop6savev2_training_adam_learning_rate_read_readvariableop%savev2_gru_kernel_read_readvariableop/savev2_gru_recurrent_kernel_read_readvariableop#savev2_gru_bias_read_readvariableop savev2_total_read_readvariableop"savev2_count_3_read_readvariableop>savev2_training_adam_features_embeddings_m_read_readvariableop7savev2_training_adam_dense_kernel_m_read_readvariableop5savev2_training_adam_dense_bias_m_read_readvariableop5savev2_training_adam_gru_kernel_m_read_readvariableop?savev2_training_adam_gru_recurrent_kernel_m_read_readvariableop3savev2_training_adam_gru_bias_m_read_readvariableop>savev2_training_adam_features_embeddings_v_read_readvariableop7savev2_training_adam_dense_kernel_v_read_readvariableop5savev2_training_adam_dense_bias_v_read_readvariableop5savev2_training_adam_gru_kernel_v_read_readvariableop?savev2_training_adam_gru_recurrent_kernel_v_read_readvariableop3savev2_training_adam_gru_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *'
dtypes
2	h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: �
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
_output_shapes
: *
T0s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :
��: :: : : : : :`: `:`: : :
��: ::`: `:`:
��: ::`: `:`: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : 
�	
�
)__inference_sequential_layer_call_fn_3665

inputs/
+statefulpartitionedcall_features_embeddings&
"statefulpartitionedcall_gru_kernel$
 statefulpartitionedcall_gru_bias0
,statefulpartitionedcall_gru_recurrent_kernel(
$statefulpartitionedcall_dense_kernel&
"statefulpartitionedcall_dense_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs+statefulpartitionedcall_features_embeddings"statefulpartitionedcall_gru_kernel statefulpartitionedcall_gru_bias,statefulpartitionedcall_gru_recurrent_kernel$statefulpartitionedcall_dense_kernel"statefulpartitionedcall_dense_bias*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_3044*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
	2*'
_output_shapes
:���������*+
_gradient_op_typePartitionedCall-3045�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*>
_input_shapes-
+:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : 
�
�
while_cond_2197
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor

gru_kernel
gru_bias
gru_recurrent_kernel
identity
P
LessLessplaceholderless_strided_slice_1*
_output_shapes
: *
T0?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*>
_input_shapes-
+: : : : :��������� : : :::: : :	 :  : : : : : : 
�d
�
 __inference__traced_restore_5193
file_prefix(
$assignvariableop_features_embeddings#
assignvariableop_1_dense_kernel!
assignvariableop_2_dense_bias)
%assignvariableop_3_training_adam_iter+
'assignvariableop_4_training_adam_beta_1+
'assignvariableop_5_training_adam_beta_2*
&assignvariableop_6_training_adam_decay2
.assignvariableop_7_training_adam_learning_rate!
assignvariableop_8_gru_kernel+
'assignvariableop_9_gru_recurrent_kernel 
assignvariableop_10_gru_bias
assignvariableop_11_total
assignvariableop_12_count_3;
7assignvariableop_13_training_adam_features_embeddings_m4
0assignvariableop_14_training_adam_dense_kernel_m2
.assignvariableop_15_training_adam_dense_bias_m2
.assignvariableop_16_training_adam_gru_kernel_m<
8assignvariableop_17_training_adam_gru_recurrent_kernel_m0
,assignvariableop_18_training_adam_gru_bias_m;
7assignvariableop_19_training_adam_features_embeddings_v4
0assignvariableop_20_training_adam_dense_kernel_v2
.assignvariableop_21_training_adam_dense_bias_v2
.assignvariableop_22_training_adam_gru_kernel_v<
8assignvariableop_23_training_adam_gru_recurrent_kernel_v0
,assignvariableop_24_training_adam_gru_bias_v
identity_26��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*�
value�B�B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE�
RestoreV2/shape_and_slicesConst"/device:CPU:0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2	L
IdentityIdentityRestoreV2:tensors:0*
_output_shapes
:*
T0�
AssignVariableOpAssignVariableOp$assignvariableop_features_embeddingsIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_kernelIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:}
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_biasIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
_output_shapes
:*
T0	�
AssignVariableOp_3AssignVariableOp%assignvariableop_3_training_adam_iterIdentity_3:output:0*
dtype0	*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
_output_shapes
:*
T0�
AssignVariableOp_4AssignVariableOp'assignvariableop_4_training_adam_beta_1Identity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp'assignvariableop_5_training_adam_beta_2Identity_5:output:0*
_output_shapes
 *
dtype0N

Identity_6IdentityRestoreV2:tensors:6*
_output_shapes
:*
T0�
AssignVariableOp_6AssignVariableOp&assignvariableop_6_training_adam_decayIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp.assignvariableop_7_training_adam_learning_rateIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:}
AssignVariableOp_8AssignVariableOpassignvariableop_8_gru_kernelIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp'assignvariableop_9_gru_recurrent_kernelIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:~
AssignVariableOp_10AssignVariableOpassignvariableop_10_gru_biasIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:{
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0*
_output_shapes
 *
dtype0P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:}
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_3Identity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp7assignvariableop_13_training_adam_features_embeddings_mIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp0assignvariableop_14_training_adam_dense_kernel_mIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp.assignvariableop_15_training_adam_dense_bias_mIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp.assignvariableop_16_training_adam_gru_kernel_mIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp8assignvariableop_17_training_adam_gru_recurrent_kernel_mIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp,assignvariableop_18_training_adam_gru_bias_mIdentity_18:output:0*
_output_shapes
 *
dtype0P
Identity_19IdentityRestoreV2:tensors:19*
_output_shapes
:*
T0�
AssignVariableOp_19AssignVariableOp7assignvariableop_19_training_adam_features_embeddings_vIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
_output_shapes
:*
T0�
AssignVariableOp_20AssignVariableOp0assignvariableop_20_training_adam_dense_kernel_vIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
_output_shapes
:*
T0�
AssignVariableOp_21AssignVariableOp.assignvariableop_21_training_adam_dense_bias_vIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
_output_shapes
:*
T0�
AssignVariableOp_22AssignVariableOp.assignvariableop_22_training_adam_gru_kernel_vIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp8assignvariableop_23_training_adam_gru_recurrent_kernel_vIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
_output_shapes
:*
T0�
AssignVariableOp_24AssignVariableOp,assignvariableop_24_training_adam_gru_bias_vIdentity_24:output:0*
dtype0*
_output_shapes
 �
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
_output_shapes
: *
T0�
Identity_26IdentityIdentity_25:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_26Identity_26:output:0*y
_input_shapesh
f: :::::::::::::::::::::::::2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112
RestoreV2_1RestoreV2_12*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_24AssignVariableOp_242$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV2: : : : :	 :
 : : : : : : : : : : : : : : : :+ '
%
_user_specified_namefile_prefix: : : : 
�
�
sequential_gru_while_cond_1562%
!sequential_gru_while_loop_counter+
'sequential_gru_while_maximum_iterations
placeholder
placeholder_1
placeholder_2'
#less_sequential_gru_strided_slice_1:
6sequential_gru_tensorarrayunstack_tensorlistfromtensor

gru_kernel
gru_bias
gru_recurrent_kernel
identity
_
LessLessplaceholder#less_sequential_gru_strided_slice_1*
_output_shapes
: *
T0?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*>
_input_shapes-
+: : : : :��������� : : :::: :	 :  : : : : : : : 
�
�
?__inference_dense_layer_call_and_return_conditional_losses_4798

inputs&
"matmul_readvariableop_dense_kernel%
!biasadd_readvariableop_dense_bias
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpx
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel*
dtype0*
_output_shapes

: i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������t
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*'
_output_shapes
:���������*
T0�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:��������� ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
�
'__inference_features_layer_call_fn_3691

inputs/
+statefulpartitionedcall_features_embeddings
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs+statefulpartitionedcall_features_embeddings**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:���������*+
_gradient_op_typePartitionedCall-2417*K
fFRD
B__inference_features_layer_call_and_return_conditional_losses_2410*
Tout
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������"
identityIdentity:output:0**
_input_shapes
:���������:22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs
�j
�
gru_while_body_3499
gru_while_loop_counter 
gru_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
gru_strided_slice_1_0U
Qtensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_gru_kernel_0
readvariableop_3_gru_bias_0+
'readvariableop_6_gru_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4
gru_strided_slice_1S
Otensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor
readvariableop_gru_kernel
readvariableop_3_gru_bias)
%readvariableop_6_gru_recurrent_kernel��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�ReadVariableOp_7�ReadVariableOp_8�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemQtensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:���������j
ReadVariableOpReadVariableOpreadvariableop_gru_kernel_0*
dtype0*
_output_shapes

:`d
strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
T0*
Index0�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice:output:0*
T0*'
_output_shapes
:��������� }
ReadVariableOp_1ReadVariableOpreadvariableop_gru_kernel_0^ReadVariableOp*
dtype0*
_output_shapes

:`f
strided_slice_1/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB"    @   h
strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

: �
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_1:output:0*'
_output_shapes
:��������� *
T0
ReadVariableOp_2ReadVariableOpreadvariableop_gru_kernel_0^ReadVariableOp_1*
_output_shapes

:`*
dtype0f
strided_slice_2/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_1Const*
_output_shapes
:*
valueB"        *
dtype0h
strided_slice_2/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
T0*
Index0�
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_2:output:0*
T0*'
_output_shapes
:��������� h
ReadVariableOp_3ReadVariableOpreadvariableop_3_gru_bias_0*
dtype0*
_output_shapes
:`_
strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*

begin_mask*
_output_shapes
: p
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*'
_output_shapes
:��������� {
ReadVariableOp_4ReadVariableOpreadvariableop_3_gru_bias_0^ReadVariableOp_3*
dtype0*
_output_shapes
:`_
strided_slice_4/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_4/stack_1Const*
valueB:@*
dtype0*
_output_shapes
:a
strided_slice_4/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_4StridedSliceReadVariableOp_4:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
_output_shapes
: *
T0*
Index0t
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*'
_output_shapes
:��������� *
T0{
ReadVariableOp_5ReadVariableOpreadvariableop_3_gru_bias_0^ReadVariableOp_4*
dtype0*
_output_shapes
:`_
strided_slice_5/stackConst*
valueB:@*
dtype0*
_output_shapes
:a
strided_slice_5/stack_1Const*
_output_shapes
:*
valueB: *
dtype0a
strided_slice_5/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
strided_slice_5StridedSliceReadVariableOp_5:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
end_mask*
_output_shapes
: *
T0*
Index0t
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:��������� x
ReadVariableOp_6ReadVariableOp'readvariableop_6_gru_recurrent_kernel_0*
dtype0*
_output_shapes

: `f
strided_slice_6/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_6/stack_1Const*
_output_shapes
:*
valueB"        *
dtype0h
strided_slice_6/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_6StridedSliceReadVariableOp_6:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
Index0*
T0m
MatMul_3MatMulplaceholder_2strided_slice_6:output:0*'
_output_shapes
:��������� *
T0�
ReadVariableOp_7ReadVariableOp'readvariableop_6_gru_recurrent_kernel_0^ReadVariableOp_6*
dtype0*
_output_shapes

: `f
strided_slice_7/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_7/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_7/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_7StridedSliceReadVariableOp_7:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:  m
MatMul_4MatMulplaceholder_2strided_slice_7:output:0*
T0*'
_output_shapes
:��������� d
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:��������� J
ConstConst*
dtype0*
_output_shapes
: *
valueB
 *��L>L
Const_1Const*
valueB
 *   ?*
dtype0*
_output_shapes
: U
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:��������� Y
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:��������� \
clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:��������� T
clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:��������� h
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:��������� L
Const_2Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_3Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_2:output:0*'
_output_shapes
:��������� *
T0[
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:��������� ^
clip_by_value_1/Minimum/yConst*
dtype0*
_output_shapes
: *
valueB
 *  �?�
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:��������� V
clip_by_value_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:��������� b
mul_2Mulclip_by_value_1:z:0placeholder_2*'
_output_shapes
:��������� *
T0�
ReadVariableOp_8ReadVariableOp'readvariableop_6_gru_recurrent_kernel_0^ReadVariableOp_7*
dtype0*
_output_shapes

: `f
strided_slice_8/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_8/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_8/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_8StridedSliceReadVariableOp_8:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
Index0*
T0i
MatMul_5MatMul	mul_2:z:0strided_slice_8:output:0*
T0*'
_output_shapes
:��������� h
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:��������� `
mul_3Mulclip_by_value:z:0placeholder_2*'
_output_shapes
:��������� *
T0J
sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: _
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:��������� Q
mul_4Mulsub:z:0Tanh:y:0*'
_output_shapes
:��������� *
T0V
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:��������� �
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_5:z:0*
element_dtype0*
_output_shapes
: I
add_6/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_6AddV2placeholderadd_6/y:output:0*
_output_shapes
: *
T0I
add_7/yConst*
value	B :*
dtype0*
_output_shapes
: Y
add_7AddV2gru_while_loop_counteradd_7/y:output:0*
_output_shapes
: *
T0�
IdentityIdentity	add_7:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: �

Identity_1Identitygru_while_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: �

Identity_2Identity	add_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
_output_shapes
: *
T0�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
_output_shapes
: *
T0�

Identity_4Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*'
_output_shapes
:��������� ",
gru_strided_slice_1gru_strided_slice_1_0"P
%readvariableop_6_gru_recurrent_kernel'readvariableop_6_gru_recurrent_kernel_0"!

identity_1Identity_1:output:0"8
readvariableop_gru_kernelreadvariableop_gru_kernel_0"8
readvariableop_3_gru_biasreadvariableop_3_gru_bias_0"�
Otensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensorQtensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*>
_input_shapes-
+: : : : :��������� : : :::2$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_5: :	 :  : : : : : : : 
�

�
)__inference_sequential_layer_call_fn_3054
features_input/
+statefulpartitionedcall_features_embeddings&
"statefulpartitionedcall_gru_kernel$
 statefulpartitionedcall_gru_bias0
,statefulpartitionedcall_gru_recurrent_kernel(
$statefulpartitionedcall_dense_kernel&
"statefulpartitionedcall_dense_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallfeatures_input+statefulpartitionedcall_features_embeddings"statefulpartitionedcall_gru_kernel statefulpartitionedcall_gru_bias,statefulpartitionedcall_gru_recurrent_kernel$statefulpartitionedcall_dense_kernel"statefulpartitionedcall_dense_bias*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_3044*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:���������*
Tin
	2*+
_gradient_op_typePartitionedCall-3045�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*>
_input_shapes-
+:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall:. *
(
_user_specified_namefeatures_input: : : : : : 
�
�
"__inference_gru_layer_call_fn_4779
inputs_0&
"statefulpartitionedcall_gru_kernel$
 statefulpartitionedcall_gru_bias0
,statefulpartitionedcall_gru_recurrent_kernel
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0"statefulpartitionedcall_gru_kernel statefulpartitionedcall_gru_bias,statefulpartitionedcall_gru_recurrent_kernel*F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_2268*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:��������� *
Tin
2*+
_gradient_op_typePartitionedCall-2269�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:��������� *
T0"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::22
StatefulPartitionedCallStatefulPartitionedCall: :( $
"
_user_specified_name
inputs/0: : 
��
�
D__inference_sequential_layer_call_and_return_conditional_losses_3376

inputs1
-features_embedding_lookup_features_embeddings!
gru_readvariableop_gru_kernel!
gru_readvariableop_3_gru_bias-
)gru_readvariableop_6_gru_recurrent_kernel,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�features/embedding_lookup�gru/ReadVariableOp�gru/ReadVariableOp_1�gru/ReadVariableOp_2�gru/ReadVariableOp_3�gru/ReadVariableOp_4�gru/ReadVariableOp_5�gru/ReadVariableOp_6�gru/ReadVariableOp_7�gru/ReadVariableOp_8�	gru/while^
features/CastCastinputs*

SrcT0*

DstT0*'
_output_shapes
:����������
features/embedding_lookupResourceGather-features_embedding_lookup_features_embeddingsfeatures/Cast:y:0*
Tindices0*
dtype0*+
_output_shapes
:���������*@
_class6
42loc:@features/embedding_lookup/features/embeddings�
"features/embedding_lookup/IdentityIdentity"features/embedding_lookup:output:0*+
_output_shapes
:���������*
T0*@
_class6
42loc:@features/embedding_lookup/features/embeddings�
$features/embedding_lookup/Identity_1Identity+features/embedding_lookup/Identity:output:0*+
_output_shapes
:���������*
T0f
	gru/ShapeShape-features/embedding_lookup/Identity_1:output:0*
_output_shapes
:*
T0a
gru/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:c
gru/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:c
gru/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: Q
gru/zeros/mul/yConst*
value	B : *
dtype0*
_output_shapes
: k
gru/zeros/mulMulgru/strided_slice:output:0gru/zeros/mul/y:output:0*
_output_shapes
: *
T0S
gru/zeros/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: e
gru/zeros/LessLessgru/zeros/mul:z:0gru/zeros/Less/y:output:0*
_output_shapes
: *
T0T
gru/zeros/packed/1Const*
value	B : *
dtype0*
_output_shapes
: 
gru/zeros/packedPackgru/strided_slice:output:0gru/zeros/packed/1:output:0*
T0*
N*
_output_shapes
:T
gru/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    x
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*'
_output_shapes
:��������� g
gru/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
gru/transpose	Transpose-features/embedding_lookup/Identity_1:output:0gru/transpose/perm:output:0*+
_output_shapes
:���������*
T0L
gru/Shape_1Shapegru/transpose:y:0*
_output_shapes
:*
T0c
gru/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB: e
gru/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:e
gru/strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
gru/strided_slice_1StridedSlicegru/Shape_1:output:0"gru/strided_slice_1/stack:output:0$gru/strided_slice_1/stack_1:output:0$gru/strided_slice_1/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0j
gru/TensorArrayV2/element_shapeConst*
valueB :
���������*
dtype0*
_output_shapes
: �
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: �
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
+gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/transpose:y:0Bgru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: c
gru/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB: e
gru/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:e
gru/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
gru/strided_slice_2StridedSlicegru/transpose:y:0"gru/strided_slice_2/stack:output:0$gru/strided_slice_2/stack_1:output:0$gru/strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:���������p
gru/ReadVariableOpReadVariableOpgru_readvariableop_gru_kernel*
dtype0*
_output_shapes

:`j
gru/strided_slice_3/stackConst*
_output_shapes
:*
valueB"        *
dtype0l
gru/strided_slice_3/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:l
gru/strided_slice_3/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
gru/strided_slice_3StridedSlicegru/ReadVariableOp:value:0"gru/strided_slice_3/stack:output:0$gru/strided_slice_3/stack_1:output:0$gru/strided_slice_3/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
T0*
Index0�

gru/MatMulMatMulgru/strided_slice_2:output:0gru/strided_slice_3:output:0*'
_output_shapes
:��������� *
T0�
gru/ReadVariableOp_1ReadVariableOpgru_readvariableop_gru_kernel^gru/ReadVariableOp*
dtype0*
_output_shapes

:`j
gru/strided_slice_4/stackConst*
valueB"        *
dtype0*
_output_shapes
:l
gru/strided_slice_4/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:l
gru/strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
gru/strided_slice_4StridedSlicegru/ReadVariableOp_1:value:0"gru/strided_slice_4/stack:output:0$gru/strided_slice_4/stack_1:output:0$gru/strided_slice_4/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
T0*
Index0�
gru/MatMul_1MatMulgru/strided_slice_2:output:0gru/strided_slice_4:output:0*'
_output_shapes
:��������� *
T0�
gru/ReadVariableOp_2ReadVariableOpgru_readvariableop_gru_kernel^gru/ReadVariableOp_1*
_output_shapes

:`*
dtype0j
gru/strided_slice_5/stackConst*
dtype0*
_output_shapes
:*
valueB"    @   l
gru/strided_slice_5/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:l
gru/strided_slice_5/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
gru/strided_slice_5StridedSlicegru/ReadVariableOp_2:value:0"gru/strided_slice_5/stack:output:0$gru/strided_slice_5/stack_1:output:0$gru/strided_slice_5/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
T0*
Index0�
gru/MatMul_2MatMulgru/strided_slice_2:output:0gru/strided_slice_5:output:0*
T0*'
_output_shapes
:��������� n
gru/ReadVariableOp_3ReadVariableOpgru_readvariableop_3_gru_bias*
dtype0*
_output_shapes
:`c
gru/strided_slice_6/stackConst*
valueB: *
dtype0*
_output_shapes
:e
gru/strided_slice_6/stack_1Const*
valueB: *
dtype0*
_output_shapes
:e
gru/strided_slice_6/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
gru/strided_slice_6StridedSlicegru/ReadVariableOp_3:value:0"gru/strided_slice_6/stack:output:0$gru/strided_slice_6/stack_1:output:0$gru/strided_slice_6/stack_2:output:0*

begin_mask*
_output_shapes
: *
T0*
Index0|
gru/BiasAddBiasAddgru/MatMul:product:0gru/strided_slice_6:output:0*'
_output_shapes
:��������� *
T0�
gru/ReadVariableOp_4ReadVariableOpgru_readvariableop_3_gru_bias^gru/ReadVariableOp_3*
dtype0*
_output_shapes
:`c
gru/strided_slice_7/stackConst*
valueB: *
dtype0*
_output_shapes
:e
gru/strided_slice_7/stack_1Const*
valueB:@*
dtype0*
_output_shapes
:e
gru/strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
gru/strided_slice_7StridedSlicegru/ReadVariableOp_4:value:0"gru/strided_slice_7/stack:output:0$gru/strided_slice_7/stack_1:output:0$gru/strided_slice_7/stack_2:output:0*
_output_shapes
: *
T0*
Index0�
gru/BiasAdd_1BiasAddgru/MatMul_1:product:0gru/strided_slice_7:output:0*
T0*'
_output_shapes
:��������� �
gru/ReadVariableOp_5ReadVariableOpgru_readvariableop_3_gru_bias^gru/ReadVariableOp_4*
dtype0*
_output_shapes
:`c
gru/strided_slice_8/stackConst*
valueB:@*
dtype0*
_output_shapes
:e
gru/strided_slice_8/stack_1Const*
valueB: *
dtype0*
_output_shapes
:e
gru/strided_slice_8/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
gru/strided_slice_8StridedSlicegru/ReadVariableOp_5:value:0"gru/strided_slice_8/stack:output:0$gru/strided_slice_8/stack_1:output:0$gru/strided_slice_8/stack_2:output:0*
end_mask*
_output_shapes
: *
T0*
Index0�
gru/BiasAdd_2BiasAddgru/MatMul_2:product:0gru/strided_slice_8:output:0*
T0*'
_output_shapes
:��������� ~
gru/ReadVariableOp_6ReadVariableOp)gru_readvariableop_6_gru_recurrent_kernel*
dtype0*
_output_shapes

: `j
gru/strided_slice_9/stackConst*
valueB"        *
dtype0*
_output_shapes
:l
gru/strided_slice_9/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:l
gru/strided_slice_9/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
gru/strided_slice_9StridedSlicegru/ReadVariableOp_6:value:0"gru/strided_slice_9/stack:output:0$gru/strided_slice_9/stack_1:output:0$gru/strided_slice_9/stack_2:output:0*
_output_shapes

:  *
T0*
Index0*

begin_mask*
end_maskz
gru/MatMul_3MatMulgru/zeros:output:0gru/strided_slice_9:output:0*
T0*'
_output_shapes
:��������� �
gru/ReadVariableOp_7ReadVariableOp)gru_readvariableop_6_gru_recurrent_kernel^gru/ReadVariableOp_6*
dtype0*
_output_shapes

: `k
gru/strided_slice_10/stackConst*
valueB"        *
dtype0*
_output_shapes
:m
gru/strided_slice_10/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:m
gru/strided_slice_10/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
gru/strided_slice_10StridedSlicegru/ReadVariableOp_7:value:0#gru/strided_slice_10/stack:output:0%gru/strided_slice_10/stack_1:output:0%gru/strided_slice_10/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:  {
gru/MatMul_4MatMulgru/zeros:output:0gru/strided_slice_10:output:0*
T0*'
_output_shapes
:��������� p
gru/addAddV2gru/BiasAdd:output:0gru/MatMul_3:product:0*'
_output_shapes
:��������� *
T0N
	gru/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *��L>P
gru/Const_1Const*
valueB
 *   ?*
dtype0*
_output_shapes
: a
gru/MulMulgru/add:z:0gru/Const:output:0*
T0*'
_output_shapes
:��������� e
	gru/Add_1Addgru/Mul:z:0gru/Const_1:output:0*'
_output_shapes
:��������� *
T0`
gru/clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
gru/clip_by_value/MinimumMinimumgru/Add_1:z:0$gru/clip_by_value/Minimum/y:output:0*'
_output_shapes
:��������� *
T0X
gru/clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
gru/clip_by_valueMaximumgru/clip_by_value/Minimum:z:0gru/clip_by_value/y:output:0*
T0*'
_output_shapes
:��������� t
	gru/add_2AddV2gru/BiasAdd_1:output:0gru/MatMul_4:product:0*'
_output_shapes
:��������� *
T0P
gru/Const_2Const*
valueB
 *��L>*
dtype0*
_output_shapes
: P
gru/Const_3Const*
dtype0*
_output_shapes
: *
valueB
 *   ?g
	gru/Mul_1Mulgru/add_2:z:0gru/Const_2:output:0*
T0*'
_output_shapes
:��������� g
	gru/Add_3Addgru/Mul_1:z:0gru/Const_3:output:0*'
_output_shapes
:��������� *
T0b
gru/clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
gru/clip_by_value_1/MinimumMinimumgru/Add_3:z:0&gru/clip_by_value_1/Minimum/y:output:0*'
_output_shapes
:��������� *
T0Z
gru/clip_by_value_1/yConst*
_output_shapes
: *
valueB
 *    *
dtype0�
gru/clip_by_value_1Maximumgru/clip_by_value_1/Minimum:z:0gru/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:��������� o
	gru/mul_2Mulgru/clip_by_value_1:z:0gru/zeros:output:0*
T0*'
_output_shapes
:��������� �
gru/ReadVariableOp_8ReadVariableOp)gru_readvariableop_6_gru_recurrent_kernel^gru/ReadVariableOp_7*
dtype0*
_output_shapes

: `k
gru/strided_slice_11/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:m
gru/strided_slice_11/stack_1Const*
_output_shapes
:*
valueB"        *
dtype0m
gru/strided_slice_11/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
gru/strided_slice_11StridedSlicegru/ReadVariableOp_8:value:0#gru/strided_slice_11/stack:output:0%gru/strided_slice_11/stack_1:output:0%gru/strided_slice_11/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
T0*
Index0v
gru/MatMul_5MatMulgru/mul_2:z:0gru/strided_slice_11:output:0*'
_output_shapes
:��������� *
T0t
	gru/add_4AddV2gru/BiasAdd_2:output:0gru/MatMul_5:product:0*
T0*'
_output_shapes
:��������� Q
gru/TanhTanhgru/add_4:z:0*
T0*'
_output_shapes
:��������� m
	gru/mul_3Mulgru/clip_by_value:z:0gru/zeros:output:0*
T0*'
_output_shapes
:��������� N
	gru/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: k
gru/subSubgru/sub/x:output:0gru/clip_by_value:z:0*
T0*'
_output_shapes
:��������� ]
	gru/mul_4Mulgru/sub:z:0gru/Tanh:y:0*
T0*'
_output_shapes
:��������� b
	gru/add_5AddV2gru/mul_3:z:0gru/mul_4:z:0*
T0*'
_output_shapes
:��������� r
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
valueB"����    *
dtype0�
gru/TensorArrayV2_1TensorListReserve*gru/TensorArrayV2_1/element_shape:output:0gru/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: J
gru/timeConst*
value	B : *
dtype0*
_output_shapes
: g
gru/while/maximum_iterationsConst*
dtype0*
_output_shapes
: *
valueB :
���������X
gru/while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: �
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0gru/zeros:output:0gru/strided_slice_1:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_readvariableop_gru_kernelgru_readvariableop_3_gru_bias)gru_readvariableop_6_gru_recurrent_kernel^gru/ReadVariableOp_2^gru/ReadVariableOp_5^gru/ReadVariableOp_8*
condR
gru_while_cond_3220*
_num_original_outputs
*
bodyR
gru_while_body_3221*9
_output_shapes'
%: : : : :��������� : : : : : *
T
2
*8
output_shapes'
%: : : : :��������� : : : : : *
_lower_using_switch_merge(*
parallel_iterations S
gru/while/IdentityIdentitygru/while:output:0*
_output_shapes
: *
T0U
gru/while/Identity_1Identitygru/while:output:1*
T0*
_output_shapes
: U
gru/while/Identity_2Identitygru/while:output:2*
_output_shapes
: *
T0U
gru/while/Identity_3Identitygru/while:output:3*
T0*
_output_shapes
: f
gru/while/Identity_4Identitygru/while:output:4*'
_output_shapes
:��������� *
T0U
gru/while/Identity_5Identitygru/while:output:5*
T0*
_output_shapes
: U
gru/while/Identity_6Identitygru/while:output:6*
_output_shapes
: *
T0U
gru/while/Identity_7Identitygru/while:output:7*
T0*
_output_shapes
: U
gru/while/Identity_8Identitygru/while:output:8*
T0*
_output_shapes
: U
gru/while/Identity_9Identitygru/while:output:9*
_output_shapes
: *
T0�
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"����    *
dtype0*
_output_shapes
:�
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while/Identity_3:output:0=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:��������� m
gru/strided_slice_12/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:f
gru/strided_slice_12/stack_1Const*
valueB: *
dtype0*
_output_shapes
:f
gru/strided_slice_12/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
gru/strided_slice_12StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0#gru/strided_slice_12/stack:output:0%gru/strided_slice_12/stack_1:output:0%gru/strided_slice_12/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:��������� *
Index0*
T0i
gru/transpose_1/permConst*
_output_shapes
:*!
valueB"          *
dtype0�
gru/transpose_1	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_1/perm:output:0*+
_output_shapes
:��������� *
T0�
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel*
dtype0*
_output_shapes

: �
dense/MatMulMatMulgru/strided_slice_12:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
dtype0*
_output_shapes
:�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentitydense/Relu:activations:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^features/embedding_lookup^gru/ReadVariableOp^gru/ReadVariableOp_1^gru/ReadVariableOp_2^gru/ReadVariableOp_3^gru/ReadVariableOp_4^gru/ReadVariableOp_5^gru/ReadVariableOp_6^gru/ReadVariableOp_7^gru/ReadVariableOp_8
^gru/while*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*>
_input_shapes-
+:���������::::::2
	gru/while	gru/while26
features/embedding_lookupfeatures/embedding_lookup2,
gru/ReadVariableOp_1gru/ReadVariableOp_12,
gru/ReadVariableOp_2gru/ReadVariableOp_22,
gru/ReadVariableOp_3gru/ReadVariableOp_32,
gru/ReadVariableOp_4gru/ReadVariableOp_42,
gru/ReadVariableOp_5gru/ReadVariableOp_52(
gru/ReadVariableOpgru/ReadVariableOp2,
gru/ReadVariableOp_6gru/ReadVariableOp_62,
gru/ReadVariableOp_7gru/ReadVariableOp_72,
gru/ReadVariableOp_8gru/ReadVariableOp_82:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp: : : : : :& "
 
_user_specified_nameinputs: 
��
�
D__inference_sequential_layer_call_and_return_conditional_losses_3654

inputs1
-features_embedding_lookup_features_embeddings!
gru_readvariableop_gru_kernel!
gru_readvariableop_3_gru_bias-
)gru_readvariableop_6_gru_recurrent_kernel,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�features/embedding_lookup�gru/ReadVariableOp�gru/ReadVariableOp_1�gru/ReadVariableOp_2�gru/ReadVariableOp_3�gru/ReadVariableOp_4�gru/ReadVariableOp_5�gru/ReadVariableOp_6�gru/ReadVariableOp_7�gru/ReadVariableOp_8�	gru/while^
features/CastCastinputs*

SrcT0*

DstT0*'
_output_shapes
:����������
features/embedding_lookupResourceGather-features_embedding_lookup_features_embeddingsfeatures/Cast:y:0*
dtype0*+
_output_shapes
:���������*@
_class6
42loc:@features/embedding_lookup/features/embeddings*
Tindices0�
"features/embedding_lookup/IdentityIdentity"features/embedding_lookup:output:0*
T0*@
_class6
42loc:@features/embedding_lookup/features/embeddings*+
_output_shapes
:����������
$features/embedding_lookup/Identity_1Identity+features/embedding_lookup/Identity:output:0*+
_output_shapes
:���������*
T0f
	gru/ShapeShape-features/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:a
gru/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:c
gru/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:c
gru/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: Q
gru/zeros/mul/yConst*
value	B : *
dtype0*
_output_shapes
: k
gru/zeros/mulMulgru/strided_slice:output:0gru/zeros/mul/y:output:0*
T0*
_output_shapes
: S
gru/zeros/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: e
gru/zeros/LessLessgru/zeros/mul:z:0gru/zeros/Less/y:output:0*
_output_shapes
: *
T0T
gru/zeros/packed/1Const*
value	B : *
dtype0*
_output_shapes
: 
gru/zeros/packedPackgru/strided_slice:output:0gru/zeros/packed/1:output:0*
T0*
N*
_output_shapes
:T
gru/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0x
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*'
_output_shapes
:��������� g
gru/transpose/permConst*
dtype0*
_output_shapes
:*!
valueB"          �
gru/transpose	Transpose-features/embedding_lookup/Identity_1:output:0gru/transpose/perm:output:0*+
_output_shapes
:���������*
T0L
gru/Shape_1Shapegru/transpose:y:0*
T0*
_output_shapes
:c
gru/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB: e
gru/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:e
gru/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
gru/strided_slice_1StridedSlicegru/Shape_1:output:0"gru/strided_slice_1/stack:output:0$gru/strided_slice_1/stack_1:output:0$gru/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: j
gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
valueB :
���������*
dtype0�
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: �
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
+gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/transpose:y:0Bgru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: c
gru/strided_slice_2/stackConst*
_output_shapes
:*
valueB: *
dtype0e
gru/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:e
gru/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
gru/strided_slice_2StridedSlicegru/transpose:y:0"gru/strided_slice_2/stack:output:0$gru/strided_slice_2/stack_1:output:0$gru/strided_slice_2/stack_2:output:0*'
_output_shapes
:���������*
T0*
Index0*
shrink_axis_maskp
gru/ReadVariableOpReadVariableOpgru_readvariableop_gru_kernel*
dtype0*
_output_shapes

:`j
gru/strided_slice_3/stackConst*
valueB"        *
dtype0*
_output_shapes
:l
gru/strided_slice_3/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:l
gru/strided_slice_3/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
gru/strided_slice_3StridedSlicegru/ReadVariableOp:value:0"gru/strided_slice_3/stack:output:0$gru/strided_slice_3/stack_1:output:0$gru/strided_slice_3/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

: �

gru/MatMulMatMulgru/strided_slice_2:output:0gru/strided_slice_3:output:0*
T0*'
_output_shapes
:��������� �
gru/ReadVariableOp_1ReadVariableOpgru_readvariableop_gru_kernel^gru/ReadVariableOp*
_output_shapes

:`*
dtype0j
gru/strided_slice_4/stackConst*
valueB"        *
dtype0*
_output_shapes
:l
gru/strided_slice_4/stack_1Const*
_output_shapes
:*
valueB"    @   *
dtype0l
gru/strided_slice_4/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
gru/strided_slice_4StridedSlicegru/ReadVariableOp_1:value:0"gru/strided_slice_4/stack:output:0$gru/strided_slice_4/stack_1:output:0$gru/strided_slice_4/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
Index0*
T0�
gru/MatMul_1MatMulgru/strided_slice_2:output:0gru/strided_slice_4:output:0*'
_output_shapes
:��������� *
T0�
gru/ReadVariableOp_2ReadVariableOpgru_readvariableop_gru_kernel^gru/ReadVariableOp_1*
dtype0*
_output_shapes

:`j
gru/strided_slice_5/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:l
gru/strided_slice_5/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:l
gru/strided_slice_5/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
gru/strided_slice_5StridedSlicegru/ReadVariableOp_2:value:0"gru/strided_slice_5/stack:output:0$gru/strided_slice_5/stack_1:output:0$gru/strided_slice_5/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

: �
gru/MatMul_2MatMulgru/strided_slice_2:output:0gru/strided_slice_5:output:0*
T0*'
_output_shapes
:��������� n
gru/ReadVariableOp_3ReadVariableOpgru_readvariableop_3_gru_bias*
dtype0*
_output_shapes
:`c
gru/strided_slice_6/stackConst*
valueB: *
dtype0*
_output_shapes
:e
gru/strided_slice_6/stack_1Const*
valueB: *
dtype0*
_output_shapes
:e
gru/strided_slice_6/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
gru/strided_slice_6StridedSlicegru/ReadVariableOp_3:value:0"gru/strided_slice_6/stack:output:0$gru/strided_slice_6/stack_1:output:0$gru/strided_slice_6/stack_2:output:0*
_output_shapes
: *
T0*
Index0*

begin_mask|
gru/BiasAddBiasAddgru/MatMul:product:0gru/strided_slice_6:output:0*'
_output_shapes
:��������� *
T0�
gru/ReadVariableOp_4ReadVariableOpgru_readvariableop_3_gru_bias^gru/ReadVariableOp_3*
_output_shapes
:`*
dtype0c
gru/strided_slice_7/stackConst*
valueB: *
dtype0*
_output_shapes
:e
gru/strided_slice_7/stack_1Const*
valueB:@*
dtype0*
_output_shapes
:e
gru/strided_slice_7/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
gru/strided_slice_7StridedSlicegru/ReadVariableOp_4:value:0"gru/strided_slice_7/stack:output:0$gru/strided_slice_7/stack_1:output:0$gru/strided_slice_7/stack_2:output:0*
_output_shapes
: *
Index0*
T0�
gru/BiasAdd_1BiasAddgru/MatMul_1:product:0gru/strided_slice_7:output:0*'
_output_shapes
:��������� *
T0�
gru/ReadVariableOp_5ReadVariableOpgru_readvariableop_3_gru_bias^gru/ReadVariableOp_4*
dtype0*
_output_shapes
:`c
gru/strided_slice_8/stackConst*
valueB:@*
dtype0*
_output_shapes
:e
gru/strided_slice_8/stack_1Const*
valueB: *
dtype0*
_output_shapes
:e
gru/strided_slice_8/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
gru/strided_slice_8StridedSlicegru/ReadVariableOp_5:value:0"gru/strided_slice_8/stack:output:0$gru/strided_slice_8/stack_1:output:0$gru/strided_slice_8/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
end_mask�
gru/BiasAdd_2BiasAddgru/MatMul_2:product:0gru/strided_slice_8:output:0*'
_output_shapes
:��������� *
T0~
gru/ReadVariableOp_6ReadVariableOp)gru_readvariableop_6_gru_recurrent_kernel*
dtype0*
_output_shapes

: `j
gru/strided_slice_9/stackConst*
valueB"        *
dtype0*
_output_shapes
:l
gru/strided_slice_9/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:l
gru/strided_slice_9/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
gru/strided_slice_9StridedSlicegru/ReadVariableOp_6:value:0"gru/strided_slice_9/stack:output:0$gru/strided_slice_9/stack_1:output:0$gru/strided_slice_9/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
T0*
Index0z
gru/MatMul_3MatMulgru/zeros:output:0gru/strided_slice_9:output:0*'
_output_shapes
:��������� *
T0�
gru/ReadVariableOp_7ReadVariableOp)gru_readvariableop_6_gru_recurrent_kernel^gru/ReadVariableOp_6*
dtype0*
_output_shapes

: `k
gru/strided_slice_10/stackConst*
dtype0*
_output_shapes
:*
valueB"        m
gru/strided_slice_10/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:m
gru/strided_slice_10/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
gru/strided_slice_10StridedSlicegru/ReadVariableOp_7:value:0#gru/strided_slice_10/stack:output:0%gru/strided_slice_10/stack_1:output:0%gru/strided_slice_10/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:  {
gru/MatMul_4MatMulgru/zeros:output:0gru/strided_slice_10:output:0*
T0*'
_output_shapes
:��������� p
gru/addAddV2gru/BiasAdd:output:0gru/MatMul_3:product:0*
T0*'
_output_shapes
:��������� N
	gru/ConstConst*
valueB
 *��L>*
dtype0*
_output_shapes
: P
gru/Const_1Const*
valueB
 *   ?*
dtype0*
_output_shapes
: a
gru/MulMulgru/add:z:0gru/Const:output:0*
T0*'
_output_shapes
:��������� e
	gru/Add_1Addgru/Mul:z:0gru/Const_1:output:0*'
_output_shapes
:��������� *
T0`
gru/clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
gru/clip_by_value/MinimumMinimumgru/Add_1:z:0$gru/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:��������� X
gru/clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
gru/clip_by_valueMaximumgru/clip_by_value/Minimum:z:0gru/clip_by_value/y:output:0*
T0*'
_output_shapes
:��������� t
	gru/add_2AddV2gru/BiasAdd_1:output:0gru/MatMul_4:product:0*'
_output_shapes
:��������� *
T0P
gru/Const_2Const*
valueB
 *��L>*
dtype0*
_output_shapes
: P
gru/Const_3Const*
_output_shapes
: *
valueB
 *   ?*
dtype0g
	gru/Mul_1Mulgru/add_2:z:0gru/Const_2:output:0*
T0*'
_output_shapes
:��������� g
	gru/Add_3Addgru/Mul_1:z:0gru/Const_3:output:0*'
_output_shapes
:��������� *
T0b
gru/clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
gru/clip_by_value_1/MinimumMinimumgru/Add_3:z:0&gru/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:��������� Z
gru/clip_by_value_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *    �
gru/clip_by_value_1Maximumgru/clip_by_value_1/Minimum:z:0gru/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:��������� o
	gru/mul_2Mulgru/clip_by_value_1:z:0gru/zeros:output:0*
T0*'
_output_shapes
:��������� �
gru/ReadVariableOp_8ReadVariableOp)gru_readvariableop_6_gru_recurrent_kernel^gru/ReadVariableOp_7*
dtype0*
_output_shapes

: `k
gru/strided_slice_11/stackConst*
_output_shapes
:*
valueB"    @   *
dtype0m
gru/strided_slice_11/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:m
gru/strided_slice_11/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
gru/strided_slice_11StridedSlicegru/ReadVariableOp_8:value:0#gru/strided_slice_11/stack:output:0%gru/strided_slice_11/stack_1:output:0%gru/strided_slice_11/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
Index0*
T0v
gru/MatMul_5MatMulgru/mul_2:z:0gru/strided_slice_11:output:0*
T0*'
_output_shapes
:��������� t
	gru/add_4AddV2gru/BiasAdd_2:output:0gru/MatMul_5:product:0*'
_output_shapes
:��������� *
T0Q
gru/TanhTanhgru/add_4:z:0*
T0*'
_output_shapes
:��������� m
	gru/mul_3Mulgru/clip_by_value:z:0gru/zeros:output:0*
T0*'
_output_shapes
:��������� N
	gru/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: k
gru/subSubgru/sub/x:output:0gru/clip_by_value:z:0*'
_output_shapes
:��������� *
T0]
	gru/mul_4Mulgru/sub:z:0gru/Tanh:y:0*'
_output_shapes
:��������� *
T0b
	gru/add_5AddV2gru/mul_3:z:0gru/mul_4:z:0*
T0*'
_output_shapes
:��������� r
!gru/TensorArrayV2_1/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"����    �
gru/TensorArrayV2_1TensorListReserve*gru/TensorArrayV2_1/element_shape:output:0gru/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: J
gru/timeConst*
value	B : *
dtype0*
_output_shapes
: g
gru/while/maximum_iterationsConst*
valueB :
���������*
dtype0*
_output_shapes
: X
gru/while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: �
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0gru/zeros:output:0gru/strided_slice_1:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_readvariableop_gru_kernelgru_readvariableop_3_gru_bias)gru_readvariableop_6_gru_recurrent_kernel^gru/ReadVariableOp_2^gru/ReadVariableOp_5^gru/ReadVariableOp_8*
T
2
*8
output_shapes'
%: : : : :��������� : : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
gru_while_cond_3498*
_num_original_outputs
*
bodyR
gru_while_body_3499*9
_output_shapes'
%: : : : :��������� : : : : : S
gru/while/IdentityIdentitygru/while:output:0*
T0*
_output_shapes
: U
gru/while/Identity_1Identitygru/while:output:1*
T0*
_output_shapes
: U
gru/while/Identity_2Identitygru/while:output:2*
T0*
_output_shapes
: U
gru/while/Identity_3Identitygru/while:output:3*
T0*
_output_shapes
: f
gru/while/Identity_4Identitygru/while:output:4*
T0*'
_output_shapes
:��������� U
gru/while/Identity_5Identitygru/while:output:5*
T0*
_output_shapes
: U
gru/while/Identity_6Identitygru/while:output:6*
T0*
_output_shapes
: U
gru/while/Identity_7Identitygru/while:output:7*
_output_shapes
: *
T0U
gru/while/Identity_8Identitygru/while:output:8*
_output_shapes
: *
T0U
gru/while/Identity_9Identitygru/while:output:9*
_output_shapes
: *
T0�
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
valueB"����    *
dtype0�
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while/Identity_3:output:0=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:��������� m
gru/strided_slice_12/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:f
gru/strided_slice_12/stack_1Const*
_output_shapes
:*
valueB: *
dtype0f
gru/strided_slice_12/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
gru/strided_slice_12StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0#gru/strided_slice_12/stack:output:0%gru/strided_slice_12/stack_1:output:0%gru/strided_slice_12/stack_2:output:0*'
_output_shapes
:��������� *
Index0*
T0*
shrink_axis_maski
gru/transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
gru/transpose_1	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� �
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel*
_output_shapes

: *
dtype0�
dense/MatMulMatMulgru/strided_slice_12:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
dtype0*
_output_shapes
:�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentitydense/Relu:activations:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^features/embedding_lookup^gru/ReadVariableOp^gru/ReadVariableOp_1^gru/ReadVariableOp_2^gru/ReadVariableOp_3^gru/ReadVariableOp_4^gru/ReadVariableOp_5^gru/ReadVariableOp_6^gru/ReadVariableOp_7^gru/ReadVariableOp_8
^gru/while*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*>
_input_shapes-
+:���������::::::2,
gru/ReadVariableOp_1gru/ReadVariableOp_12,
gru/ReadVariableOp_2gru/ReadVariableOp_22,
gru/ReadVariableOp_3gru/ReadVariableOp_32,
gru/ReadVariableOp_4gru/ReadVariableOp_42(
gru/ReadVariableOpgru/ReadVariableOp2,
gru/ReadVariableOp_5gru/ReadVariableOp_52,
gru/ReadVariableOp_6gru/ReadVariableOp_62,
gru/ReadVariableOp_7gru/ReadVariableOp_72,
gru/ReadVariableOp_8gru/ReadVariableOp_82<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2
	gru/while	gru/while26
features/embedding_lookupfeatures/embedding_lookup:& "
 
_user_specified_nameinputs: : : : : : 
�:
�
=__inference_gru_layer_call_and_return_conditional_losses_2268

inputs&
"statefulpartitionedcall_gru_kernel$
 statefulpartitionedcall_gru_bias0
,statefulpartitionedcall_gru_recurrent_kernel
identity��StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B : *
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B : *
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*'
_output_shapes
:��������� *
T0c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
valueB: *
dtype0a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
valueB :
���������*
dtype0�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: �
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
element_dtype0*
_output_shapes
: *

shape_type0_
strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
_output_shapes
:*
valueB:*
dtype0a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:���������*
T0*
Index0�
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0"statefulpartitionedcall_gru_kernel statefulpartitionedcall_gru_bias,statefulpartitionedcall_gru_recurrent_kernel**
config_proto

GPU 

CPU2J 8*:
_output_shapes(
&:��������� :��������� *
Tin	
2*+
_gradient_op_typePartitionedCall-1939*K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_1842*
Tout
2n
TensorArrayV2_1/element_shapeConst*
valueB"����    *
dtype0*
_output_shapes
:�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
valueB :
���������*
dtype0*
_output_shapes
: T
while/loop_counterConst*
_output_shapes
: *
value	B : *
dtype0�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"statefulpartitionedcall_gru_kernel statefulpartitionedcall_gru_bias,statefulpartitionedcall_gru_recurrent_kernel^StatefulPartitionedCall*
T
2
*8
output_shapes'
%: : : : :��������� : : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_2197*
_num_original_outputs
*
bodyR
while_body_2198*9
_output_shapes'
%: : : : :��������� : : : : : K
while/IdentityIdentitywhile:output:0*
_output_shapes
: *
T0M
while/Identity_1Identitywhile:output:1*
_output_shapes
: *
T0M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
_output_shapes
: *
T0^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:��������� M
while/Identity_5Identitywhile:output:5*
T0*
_output_shapes
: M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
_output_shapes
: *
T0M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
_output_shapes
: *
T0�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"����    *
dtype0*
_output_shapes
:�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :������������������ h
strided_slice_3/stackConst*
_output_shapes
:*
valueB:
���������*
dtype0a
strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:��������� *
T0*
Index0e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ �
IdentityIdentitystrided_slice_3:output:0^StatefulPartitionedCall^while*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*?
_input_shapes.
,:������������������:::2
whilewhile22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
��
�
=__inference_gru_layer_call_and_return_conditional_losses_4771
inputs_0
readvariableop_gru_kernel
readvariableop_3_gru_bias)
%readvariableop_6_gru_recurrent_kernel
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�ReadVariableOp_7�ReadVariableOp_8�while=
ShapeShapeinputs_0*
_output_shapes
:*
T0]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0M
zeros/mul/yConst*
dtype0*
_output_shapes
: *
value	B : _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
value
B :�*
dtype0Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
_output_shapes
: *
value	B : *
dtype0s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
_output_shapes
:*
T0P
zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0l
zerosFillzeros/packed:output:0zeros/Const:output:0*'
_output_shapes
:��������� *
T0c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
���������*
dtype0*
_output_shapes
: �
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: �
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:a
strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:���������h
ReadVariableOpReadVariableOpreadvariableop_gru_kernel*
dtype0*
_output_shapes

:`f
strided_slice_3/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_3/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_3/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_3StridedSliceReadVariableOp:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
T0*
Index0v
MatMulMatMulstrided_slice_2:output:0strided_slice_3:output:0*
T0*'
_output_shapes
:��������� {
ReadVariableOp_1ReadVariableOpreadvariableop_gru_kernel^ReadVariableOp*
dtype0*
_output_shapes

:`f
strided_slice_4/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_4/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_4/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
strided_slice_4StridedSliceReadVariableOp_1:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
T0*
Index0x
MatMul_1MatMulstrided_slice_2:output:0strided_slice_4:output:0*
T0*'
_output_shapes
:��������� }
ReadVariableOp_2ReadVariableOpreadvariableop_gru_kernel^ReadVariableOp_1*
dtype0*
_output_shapes

:`f
strided_slice_5/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_5/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_5/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_5StridedSliceReadVariableOp_2:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
Index0*
T0x
MatMul_2MatMulstrided_slice_2:output:0strided_slice_5:output:0*'
_output_shapes
:��������� *
T0f
ReadVariableOp_3ReadVariableOpreadvariableop_3_gru_bias*
dtype0*
_output_shapes
:`_
strided_slice_6/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_6/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_6/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
strided_slice_6StridedSliceReadVariableOp_3:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*

begin_mask*
_output_shapes
: *
T0*
Index0p
BiasAddBiasAddMatMul:product:0strided_slice_6:output:0*
T0*'
_output_shapes
:��������� y
ReadVariableOp_4ReadVariableOpreadvariableop_3_gru_bias^ReadVariableOp_3*
dtype0*
_output_shapes
:`_
strided_slice_7/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_7/stack_1Const*
valueB:@*
dtype0*
_output_shapes
:a
strided_slice_7/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
strided_slice_7StridedSliceReadVariableOp_4:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
_output_shapes
: *
T0*
Index0t
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_7:output:0*
T0*'
_output_shapes
:��������� y
ReadVariableOp_5ReadVariableOpreadvariableop_3_gru_bias^ReadVariableOp_4*
_output_shapes
:`*
dtype0_
strided_slice_8/stackConst*
valueB:@*
dtype0*
_output_shapes
:a
strided_slice_8/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_8/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_8StridedSliceReadVariableOp_5:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
end_mask*
_output_shapes
: *
T0*
Index0t
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_8:output:0*
T0*'
_output_shapes
:��������� v
ReadVariableOp_6ReadVariableOp%readvariableop_6_gru_recurrent_kernel*
_output_shapes

: `*
dtype0f
strided_slice_9/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_9/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_9/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_9StridedSliceReadVariableOp_6:value:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:  n
MatMul_3MatMulzeros:output:0strided_slice_9:output:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_7ReadVariableOp%readvariableop_6_gru_recurrent_kernel^ReadVariableOp_6*
dtype0*
_output_shapes

: `g
strided_slice_10/stackConst*
valueB"        *
dtype0*
_output_shapes
:i
strided_slice_10/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:i
strided_slice_10/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_10StridedSliceReadVariableOp_7:value:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:  o
MatMul_4MatMulzeros:output:0strided_slice_10:output:0*
T0*'
_output_shapes
:��������� d
addAddV2BiasAdd:output:0MatMul_3:product:0*'
_output_shapes
:��������� *
T0J
ConstConst*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *   ?U
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:��������� Y
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:��������� \
clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:��������� T
clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:��������� h
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*'
_output_shapes
:��������� *
T0L
Const_2Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_3Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_2:output:0*'
_output_shapes
:��������� *
T0[
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:��������� ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
valueB
 *  �?*
dtype0�
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*'
_output_shapes
:��������� *
T0V
clip_by_value_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*'
_output_shapes
:��������� *
T0c
mul_2Mulclip_by_value_1:z:0zeros:output:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_8ReadVariableOp%readvariableop_6_gru_recurrent_kernel^ReadVariableOp_7*
dtype0*
_output_shapes

: `g
strided_slice_11/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:i
strided_slice_11/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:i
strided_slice_11/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_11StridedSliceReadVariableOp_8:value:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
Index0*
T0j
MatMul_5MatMul	mul_2:z:0strided_slice_11:output:0*
T0*'
_output_shapes
:��������� h
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_4:z:0*'
_output_shapes
:��������� *
T0a
mul_3Mulclip_by_value:z:0zeros:output:0*
T0*'
_output_shapes
:��������� J
sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?_
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:��������� Q
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
valueB"����    *
dtype0�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
valueB :
���������*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0readvariableop_gru_kernelreadvariableop_3_gru_bias%readvariableop_6_gru_recurrent_kernel^ReadVariableOp_2^ReadVariableOp_5^ReadVariableOp_8*
parallel_iterations *
condR
while_cond_4622*
_num_original_outputs
*
bodyR
while_body_4623*9
_output_shapes'
%: : : : :��������� : : : : : *8
output_shapes'
%: : : : :��������� : : : : : *
T
2
*
_lower_using_switch_merge(K
while/IdentityIdentitywhile:output:0*
_output_shapes
: *
T0M
while/Identity_1Identitywhile:output:1*
_output_shapes
: *
T0M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:��������� M
while/Identity_5Identitywhile:output:5*
T0*
_output_shapes
: M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
_output_shapes
: *
T0M
while/Identity_8Identitywhile:output:8*
_output_shapes
: *
T0M
while/Identity_9Identitywhile:output:9*
_output_shapes
: *
T0�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"����    *
dtype0*
_output_shapes
:�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :������������������ i
strided_slice_12/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:b
strided_slice_12/stack_1Const*
dtype0*
_output_shapes
:*
valueB: b
strided_slice_12/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_12StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:��������� *
T0*
Index0e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*4
_output_shapes"
 :������������������ *
T0�
IdentityIdentitystrided_slice_12:output:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8^while*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*?
_input_shapes.
,:������������������:::2$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82
whilewhile2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_2: : :( $
"
_user_specified_name
inputs/0: 
�:
�
=__inference_gru_layer_call_and_return_conditional_losses_2390

inputs&
"statefulpartitionedcall_gru_kernel$
 statefulpartitionedcall_gru_bias0
,statefulpartitionedcall_gru_recurrent_kernel
identity��StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0M
zeros/mul/yConst*
dtype0*
_output_shapes
: *
value	B : _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
_output_shapes
: *
value	B : *
dtype0s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
_output_shapes
:*
T0P
zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0l
zerosFillzeros/packed:output:0zeros/Const:output:0*'
_output_shapes
:��������� *
T0c
transpose/permConst*
_output_shapes
:*!
valueB"          *
dtype0v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
_output_shapes
:*
T0_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0f
TensorArrayV2/element_shapeConst*
valueB :
���������*
dtype0*
_output_shapes
: �
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: �
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
valueB"����   *
dtype0�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*'
_output_shapes
:���������*
Index0*
T0*
shrink_axis_mask�
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0"statefulpartitionedcall_gru_kernel statefulpartitionedcall_gru_bias,statefulpartitionedcall_gru_recurrent_kernel*:
_output_shapes(
&:��������� :��������� *
Tin	
2*+
_gradient_op_typePartitionedCall-1954*K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_1934*
Tout
2**
config_proto

GPU 

CPU2J 8n
TensorArrayV2_1/element_shapeConst*
valueB"����    *
dtype0*
_output_shapes
:�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
dtype0*
_output_shapes
: *
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
valueB :
���������*
dtype0T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"statefulpartitionedcall_gru_kernel statefulpartitionedcall_gru_bias,statefulpartitionedcall_gru_recurrent_kernel^StatefulPartitionedCall*
bodyR
while_body_2320*9
_output_shapes'
%: : : : :��������� : : : : : *8
output_shapes'
%: : : : :��������� : : : : : *
T
2
*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_2319*
_num_original_outputs
K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
_output_shapes
: *
T0M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:��������� M
while/Identity_5Identitywhile:output:5*
_output_shapes
: *
T0M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"����    *
dtype0*
_output_shapes
:�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :������������������ h
strided_slice_3/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:��������� *
T0*
Index0e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ �
IdentityIdentitystrided_slice_3:output:0^StatefulPartitionedCall^while*'
_output_shapes
:��������� *
T0"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::2
whilewhile22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
�j
�
while_body_2810
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_gru_kernel_0
readvariableop_3_gru_bias_0+
'readvariableop_6_gru_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
readvariableop_gru_kernel
readvariableop_3_gru_bias)
%readvariableop_6_gru_recurrent_kernel��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�ReadVariableOp_7�ReadVariableOp_8�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:���������j
ReadVariableOpReadVariableOpreadvariableop_gru_kernel_0*
dtype0*
_output_shapes

:`d
strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

: �
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice:output:0*
T0*'
_output_shapes
:��������� }
ReadVariableOp_1ReadVariableOpreadvariableop_gru_kernel_0^ReadVariableOp*
dtype0*
_output_shapes

:`f
strided_slice_2/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_2/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_2StridedSliceReadVariableOp_1:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
T0*
Index0�
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_2:output:0*
T0*'
_output_shapes
:��������� 
ReadVariableOp_2ReadVariableOpreadvariableop_gru_kernel_0^ReadVariableOp_1*
_output_shapes

:`*
dtype0f
strided_slice_3/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_3/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_3/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_3StridedSliceReadVariableOp_2:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
Index0*
T0�
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_3:output:0*
T0*'
_output_shapes
:��������� h
ReadVariableOp_3ReadVariableOpreadvariableop_3_gru_bias_0*
dtype0*
_output_shapes
:`_
strided_slice_4/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_4/stack_1Const*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_4/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_4StridedSliceReadVariableOp_3:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*

begin_mask*
_output_shapes
: p
BiasAddBiasAddMatMul:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:��������� {
ReadVariableOp_4ReadVariableOpreadvariableop_3_gru_bias_0^ReadVariableOp_3*
dtype0*
_output_shapes
:`_
strided_slice_5/stackConst*
_output_shapes
:*
valueB: *
dtype0a
strided_slice_5/stack_1Const*
valueB:@*
dtype0*
_output_shapes
:a
strided_slice_5/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
strided_slice_5StridedSliceReadVariableOp_4:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
_output_shapes
: *
T0*
Index0t
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_5:output:0*'
_output_shapes
:��������� *
T0{
ReadVariableOp_5ReadVariableOpreadvariableop_3_gru_bias_0^ReadVariableOp_4*
dtype0*
_output_shapes
:`_
strided_slice_6/stackConst*
valueB:@*
dtype0*
_output_shapes
:a
strided_slice_6/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_6/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
strided_slice_6StridedSliceReadVariableOp_5:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
end_maskt
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_6:output:0*
T0*'
_output_shapes
:��������� x
ReadVariableOp_6ReadVariableOp'readvariableop_6_gru_recurrent_kernel_0*
_output_shapes

: `*
dtype0f
strided_slice_7/stackConst*
_output_shapes
:*
valueB"        *
dtype0h
strided_slice_7/stack_1Const*
dtype0*
_output_shapes
:*
valueB"        h
strided_slice_7/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_7StridedSliceReadVariableOp_6:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:  m
MatMul_3MatMulplaceholder_2strided_slice_7:output:0*'
_output_shapes
:��������� *
T0�
ReadVariableOp_7ReadVariableOp'readvariableop_6_gru_recurrent_kernel_0^ReadVariableOp_6*
_output_shapes

: `*
dtype0f
strided_slice_8/stackConst*
_output_shapes
:*
valueB"        *
dtype0h
strided_slice_8/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_8/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_8StridedSliceReadVariableOp_7:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
end_mask*
_output_shapes

:  *
T0*
Index0*

begin_maskm
MatMul_4MatMulplaceholder_2strided_slice_8:output:0*
T0*'
_output_shapes
:��������� d
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:��������� J
ConstConst*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_1Const*
valueB
 *   ?*
dtype0*
_output_shapes
: U
MulMuladd:z:0Const:output:0*'
_output_shapes
:��������� *
T0Y
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:��������� \
clip_by_value/Minimum/yConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:��������� T
clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:��������� h
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:��������� L
Const_2Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_3Const*
dtype0*
_output_shapes
: *
valueB
 *   ?[
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:��������� [
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:��������� ^
clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:��������� V
clip_by_value_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*'
_output_shapes
:��������� *
T0b
mul_2Mulclip_by_value_1:z:0placeholder_2*
T0*'
_output_shapes
:��������� �
ReadVariableOp_8ReadVariableOp'readvariableop_6_gru_recurrent_kernel_0^ReadVariableOp_7*
dtype0*
_output_shapes

: `f
strided_slice_9/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_9/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_9/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_9StridedSliceReadVariableOp_8:value:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:  i
MatMul_5MatMul	mul_2:z:0strided_slice_9:output:0*
T0*'
_output_shapes
:��������� h
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*'
_output_shapes
:��������� *
T0I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:��������� `
mul_3Mulclip_by_value:z:0placeholder_2*
T0*'
_output_shapes
:��������� J
sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: _
subSubsub/x:output:0clip_by_value:z:0*'
_output_shapes
:��������� *
T0Q
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:��������� �
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_5:z:0*
element_dtype0*
_output_shapes
: I
add_6/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_6AddV2placeholderadd_6/y:output:0*
T0*
_output_shapes
: I
add_7/yConst*
_output_shapes
: *
value	B :*
dtype0U
add_7AddV2while_loop_counteradd_7/y:output:0*
T0*
_output_shapes
: �
IdentityIdentity	add_7:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
_output_shapes
: *
T0�

Identity_1Identitywhile_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
_output_shapes
: *
T0�

Identity_2Identity	add_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
_output_shapes
: *
T0�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
_output_shapes
: *
T0�

Identity_4Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*'
_output_shapes
:��������� *
T0"$
strided_slice_1strided_slice_1_0"P
%readvariableop_6_gru_recurrent_kernel'readvariableop_6_gru_recurrent_kernel_0"!

identity_1Identity_1:output:0"8
readvariableop_3_gru_biasreadvariableop_3_gru_bias_0"8
readvariableop_gru_kernelreadvariableop_gru_kernel_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*>
_input_shapes-
+: : : : :��������� : : :::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_8:  : : : : : : : : :	 
�
�
B__inference_features_layer_call_and_return_conditional_losses_2410

inputs(
$embedding_lookup_features_embeddings
identity��embedding_lookupU
CastCastinputs*

SrcT0*

DstT0*'
_output_shapes
:����������
embedding_lookupResourceGather$embedding_lookup_features_embeddingsCast:y:0*
Tindices0*
dtype0*+
_output_shapes
:���������*7
_class-
+)loc:@embedding_lookup/features/embeddings�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*7
_class-
+)loc:@embedding_lookup/features/embeddings*+
_output_shapes
:����������
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:����������
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*+
_output_shapes
:���������*
T0"
identityIdentity:output:0**
_input_shapes
:���������:2$
embedding_lookupembedding_lookup:& "
 
_user_specified_nameinputs: 
�j
�
gru_while_body_3221
gru_while_loop_counter 
gru_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
gru_strided_slice_1_0U
Qtensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_gru_kernel_0
readvariableop_3_gru_bias_0+
'readvariableop_6_gru_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4
gru_strided_slice_1S
Otensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor
readvariableop_gru_kernel
readvariableop_3_gru_bias)
%readvariableop_6_gru_recurrent_kernel��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�ReadVariableOp_7�ReadVariableOp_8�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"����   �
#TensorArrayV2Read/TensorListGetItemTensorListGetItemQtensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:���������j
ReadVariableOpReadVariableOpreadvariableop_gru_kernel_0*
dtype0*
_output_shapes

:`d
strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_1Const*
_output_shapes
:*
valueB"        *
dtype0f
strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
Index0*
T0�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice:output:0*
T0*'
_output_shapes
:��������� }
ReadVariableOp_1ReadVariableOpreadvariableop_gru_kernel_0^ReadVariableOp*
dtype0*
_output_shapes

:`f
strided_slice_1/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_1/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
T0*
Index0�
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_1:output:0*
T0*'
_output_shapes
:��������� 
ReadVariableOp_2ReadVariableOpreadvariableop_gru_kernel_0^ReadVariableOp_1*
dtype0*
_output_shapes

:`f
strided_slice_2/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_1Const*
_output_shapes
:*
valueB"        *
dtype0h
strided_slice_2/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
Index0*
T0�
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_2:output:0*
T0*'
_output_shapes
:��������� h
ReadVariableOp_3ReadVariableOpreadvariableop_3_gru_bias_0*
dtype0*
_output_shapes
:`_
strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*

begin_mask*
_output_shapes
: *
T0*
Index0p
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*'
_output_shapes
:��������� {
ReadVariableOp_4ReadVariableOpreadvariableop_3_gru_bias_0^ReadVariableOp_3*
dtype0*
_output_shapes
:`_
strided_slice_4/stackConst*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_4/stack_1Const*
valueB:@*
dtype0*
_output_shapes
:a
strided_slice_4/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
strided_slice_4StridedSliceReadVariableOp_4:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
_output_shapes
: *
T0*
Index0t
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:��������� {
ReadVariableOp_5ReadVariableOpreadvariableop_3_gru_bias_0^ReadVariableOp_4*
dtype0*
_output_shapes
:`_
strided_slice_5/stackConst*
valueB:@*
dtype0*
_output_shapes
:a
strided_slice_5/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_5/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_5StridedSliceReadVariableOp_5:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
end_maskt
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:��������� x
ReadVariableOp_6ReadVariableOp'readvariableop_6_gru_recurrent_kernel_0*
_output_shapes

: `*
dtype0f
strided_slice_6/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_6/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_6/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_6StridedSliceReadVariableOp_6:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:  m
MatMul_3MatMulplaceholder_2strided_slice_6:output:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_7ReadVariableOp'readvariableop_6_gru_recurrent_kernel_0^ReadVariableOp_6*
dtype0*
_output_shapes

: `f
strided_slice_7/stackConst*
dtype0*
_output_shapes
:*
valueB"        h
strided_slice_7/stack_1Const*
dtype0*
_output_shapes
:*
valueB"    @   h
strided_slice_7/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_7StridedSliceReadVariableOp_7:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:  m
MatMul_4MatMulplaceholder_2strided_slice_7:output:0*
T0*'
_output_shapes
:��������� d
addAddV2BiasAdd:output:0MatMul_3:product:0*'
_output_shapes
:��������� *
T0J
ConstConst*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_1Const*
valueB
 *   ?*
dtype0*
_output_shapes
: U
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:��������� Y
Add_1AddMul:z:0Const_1:output:0*'
_output_shapes
:��������� *
T0\
clip_by_value/Minimum/yConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*'
_output_shapes
:��������� *
T0T
clip_by_value/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:��������� h
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:��������� L
Const_2Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_3Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_2:output:0*'
_output_shapes
:��������� *
T0[
Add_3Add	Mul_1:z:0Const_3:output:0*'
_output_shapes
:��������� *
T0^
clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*'
_output_shapes
:��������� *
T0V
clip_by_value_1/yConst*
_output_shapes
: *
valueB
 *    *
dtype0�
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:��������� b
mul_2Mulclip_by_value_1:z:0placeholder_2*'
_output_shapes
:��������� *
T0�
ReadVariableOp_8ReadVariableOp'readvariableop_6_gru_recurrent_kernel_0^ReadVariableOp_7*
dtype0*
_output_shapes

: `f
strided_slice_8/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_8/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_8/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_8StridedSliceReadVariableOp_8:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:  i
MatMul_5MatMul	mul_2:z:0strided_slice_8:output:0*
T0*'
_output_shapes
:��������� h
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:��������� `
mul_3Mulclip_by_value:z:0placeholder_2*'
_output_shapes
:��������� *
T0J
sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: _
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:��������� Q
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:��������� �
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_5:z:0*
element_dtype0*
_output_shapes
: I
add_6/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_6AddV2placeholderadd_6/y:output:0*
_output_shapes
: *
T0I
add_7/yConst*
value	B :*
dtype0*
_output_shapes
: Y
add_7AddV2gru_while_loop_counteradd_7/y:output:0*
_output_shapes
: *
T0�
IdentityIdentity	add_7:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: �

Identity_1Identitygru_while_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
_output_shapes
: *
T0�

Identity_2Identity	add_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
_output_shapes
: *
T0�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
_output_shapes
: *
T0�

Identity_4Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*'
_output_shapes
:��������� *
T0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0",
gru_strided_slice_1gru_strided_slice_1_0"P
%readvariableop_6_gru_recurrent_kernel'readvariableop_6_gru_recurrent_kernel_0"8
readvariableop_gru_kernelreadvariableop_gru_kernel_0"!

identity_1Identity_1:output:0"8
readvariableop_3_gru_biasreadvariableop_3_gru_bias_0"!

identity_2Identity_2:output:0"�
Otensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensorQtensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2$
ReadVariableOp_8ReadVariableOp_82 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_7:  : : : : : : : : :	 
�j
�
while_body_4623
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_gru_kernel_0
readvariableop_3_gru_bias_0+
'readvariableop_6_gru_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
readvariableop_gru_kernel
readvariableop_3_gru_bias)
%readvariableop_6_gru_recurrent_kernel��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�ReadVariableOp_7�ReadVariableOp_8�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:���������j
ReadVariableOpReadVariableOpreadvariableop_gru_kernel_0*
dtype0*
_output_shapes

:`d
strided_slice/stackConst*
_output_shapes
:*
valueB"        *
dtype0f
strided_slice/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
Index0*
T0�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice:output:0*'
_output_shapes
:��������� *
T0}
ReadVariableOp_1ReadVariableOpreadvariableop_gru_kernel_0^ReadVariableOp*
dtype0*
_output_shapes

:`f
strided_slice_2/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_2/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_2StridedSliceReadVariableOp_1:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
T0*
Index0�
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_2:output:0*
T0*'
_output_shapes
:��������� 
ReadVariableOp_2ReadVariableOpreadvariableop_gru_kernel_0^ReadVariableOp_1*
dtype0*
_output_shapes

:`f
strided_slice_3/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_3/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_3/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_3StridedSliceReadVariableOp_2:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

: �
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_3:output:0*
T0*'
_output_shapes
:��������� h
ReadVariableOp_3ReadVariableOpreadvariableop_3_gru_bias_0*
dtype0*
_output_shapes
:`_
strided_slice_4/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_4/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_4/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_4StridedSliceReadVariableOp_3:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
_output_shapes
: *
T0*
Index0*

begin_maskp
BiasAddBiasAddMatMul:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:��������� {
ReadVariableOp_4ReadVariableOpreadvariableop_3_gru_bias_0^ReadVariableOp_3*
dtype0*
_output_shapes
:`_
strided_slice_5/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_5/stack_1Const*
valueB:@*
dtype0*
_output_shapes
:a
strided_slice_5/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_5StridedSliceReadVariableOp_4:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
_output_shapes
: *
Index0*
T0t
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:��������� {
ReadVariableOp_5ReadVariableOpreadvariableop_3_gru_bias_0^ReadVariableOp_4*
dtype0*
_output_shapes
:`_
strided_slice_6/stackConst*
valueB:@*
dtype0*
_output_shapes
:a
strided_slice_6/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_6/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
strided_slice_6StridedSliceReadVariableOp_5:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
T0*
Index0*
end_mask*
_output_shapes
: t
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_6:output:0*
T0*'
_output_shapes
:��������� x
ReadVariableOp_6ReadVariableOp'readvariableop_6_gru_recurrent_kernel_0*
_output_shapes

: `*
dtype0f
strided_slice_7/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_7/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_7/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_slice_7StridedSliceReadVariableOp_6:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
T0*
Index0m
MatMul_3MatMulplaceholder_2strided_slice_7:output:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_7ReadVariableOp'readvariableop_6_gru_recurrent_kernel_0^ReadVariableOp_6*
dtype0*
_output_shapes

: `f
strided_slice_8/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_8/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_8/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_8StridedSliceReadVariableOp_7:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
Index0*
T0m
MatMul_4MatMulplaceholder_2strided_slice_8:output:0*'
_output_shapes
:��������� *
T0d
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:��������� J
ConstConst*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_1Const*
valueB
 *   ?*
dtype0*
_output_shapes
: U
MulMuladd:z:0Const:output:0*'
_output_shapes
:��������� *
T0Y
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:��������� \
clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:��������� T
clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*'
_output_shapes
:��������� *
T0h
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:��������� L
Const_2Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_3Const*
dtype0*
_output_shapes
: *
valueB
 *   ?[
Mul_1Mul	add_2:z:0Const_2:output:0*'
_output_shapes
:��������� *
T0[
Add_3Add	Mul_1:z:0Const_3:output:0*'
_output_shapes
:��������� *
T0^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
valueB
 *  �?*
dtype0�
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*'
_output_shapes
:��������� *
T0V
clip_by_value_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*'
_output_shapes
:��������� *
T0b
mul_2Mulclip_by_value_1:z:0placeholder_2*
T0*'
_output_shapes
:��������� �
ReadVariableOp_8ReadVariableOp'readvariableop_6_gru_recurrent_kernel_0^ReadVariableOp_7*
dtype0*
_output_shapes

: `f
strided_slice_9/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_9/stack_1Const*
dtype0*
_output_shapes
:*
valueB"        h
strided_slice_9/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
strided_slice_9StridedSliceReadVariableOp_8:value:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
Index0*
T0i
MatMul_5MatMul	mul_2:z:0strided_slice_9:output:0*
T0*'
_output_shapes
:��������� h
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:��������� `
mul_3Mulclip_by_value:z:0placeholder_2*
T0*'
_output_shapes
:��������� J
sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: _
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:��������� Q
mul_4Mulsub:z:0Tanh:y:0*'
_output_shapes
:��������� *
T0V
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:��������� �
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_5:z:0*
element_dtype0*
_output_shapes
: I
add_6/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_6AddV2placeholderadd_6/y:output:0*
T0*
_output_shapes
: I
add_7/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_7AddV2while_loop_counteradd_7/y:output:0*
T0*
_output_shapes
: �
IdentityIdentity	add_7:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: �

Identity_1Identitywhile_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
_output_shapes
: *
T0�

Identity_2Identity	add_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
_output_shapes
: *
T0�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
_output_shapes
: *
T0�

Identity_4Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*'
_output_shapes
:��������� *
T0"$
strided_slice_1strided_slice_1_0"P
%readvariableop_6_gru_recurrent_kernel'readvariableop_6_gru_recurrent_kernel_0"8
readvariableop_gru_kernelreadvariableop_gru_kernel_0"8
readvariableop_3_gru_biasreadvariableop_3_gru_bias_0"!

identity_1Identity_1:output:0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*>
_input_shapes-
+: : : : :��������� : : :::2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82 
ReadVariableOpReadVariableOp:  : : : : : : : : :	 
��
�
=__inference_gru_layer_call_and_return_conditional_losses_4505
inputs_0
readvariableop_gru_kernel
readvariableop_3_gru_bias)
%readvariableop_6_gru_recurrent_kernel
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�ReadVariableOp_7�ReadVariableOp_8�while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: _
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0M
zeros/mul/yConst*
value	B : *
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
_output_shapes
: *
T0O
zeros/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B : *
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*'
_output_shapes
:��������� *
T0c
transpose/permConst*
_output_shapes
:*!
valueB"          *
dtype0x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
_output_shapes
:*
T0_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
���������*
dtype0*
_output_shapes
: �
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: �
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
_output_shapes
:*
valueB:*
dtype0a
strided_slice_2/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:���������*
T0*
Index0h
ReadVariableOpReadVariableOpreadvariableop_gru_kernel*
_output_shapes

:`*
dtype0f
strided_slice_3/stackConst*
dtype0*
_output_shapes
:*
valueB"        h
strided_slice_3/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_3/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_3StridedSliceReadVariableOp:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
T0*
Index0v
MatMulMatMulstrided_slice_2:output:0strided_slice_3:output:0*'
_output_shapes
:��������� *
T0{
ReadVariableOp_1ReadVariableOpreadvariableop_gru_kernel^ReadVariableOp*
dtype0*
_output_shapes

:`f
strided_slice_4/stackConst*
dtype0*
_output_shapes
:*
valueB"        h
strided_slice_4/stack_1Const*
_output_shapes
:*
valueB"    @   *
dtype0h
strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_4StridedSliceReadVariableOp_1:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
T0*
Index0x
MatMul_1MatMulstrided_slice_2:output:0strided_slice_4:output:0*
T0*'
_output_shapes
:��������� }
ReadVariableOp_2ReadVariableOpreadvariableop_gru_kernel^ReadVariableOp_1*
dtype0*
_output_shapes

:`f
strided_slice_5/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_5/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_5/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_5StridedSliceReadVariableOp_2:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
end_mask*
_output_shapes

: *
T0*
Index0*

begin_maskx
MatMul_2MatMulstrided_slice_2:output:0strided_slice_5:output:0*
T0*'
_output_shapes
:��������� f
ReadVariableOp_3ReadVariableOpreadvariableop_3_gru_bias*
dtype0*
_output_shapes
:`_
strided_slice_6/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_6/stack_1Const*
_output_shapes
:*
valueB: *
dtype0a
strided_slice_6/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
strided_slice_6StridedSliceReadVariableOp_3:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*

begin_mask*
_output_shapes
: p
BiasAddBiasAddMatMul:product:0strided_slice_6:output:0*
T0*'
_output_shapes
:��������� y
ReadVariableOp_4ReadVariableOpreadvariableop_3_gru_bias^ReadVariableOp_3*
dtype0*
_output_shapes
:`_
strided_slice_7/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_7/stack_1Const*
dtype0*
_output_shapes
:*
valueB:@a
strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_7StridedSliceReadVariableOp_4:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
T0*
Index0*
_output_shapes
: t
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_7:output:0*
T0*'
_output_shapes
:��������� y
ReadVariableOp_5ReadVariableOpreadvariableop_3_gru_bias^ReadVariableOp_4*
dtype0*
_output_shapes
:`_
strided_slice_8/stackConst*
dtype0*
_output_shapes
:*
valueB:@a
strided_slice_8/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_8/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_8StridedSliceReadVariableOp_5:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
end_mask*
_output_shapes
: *
Index0*
T0t
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_8:output:0*
T0*'
_output_shapes
:��������� v
ReadVariableOp_6ReadVariableOp%readvariableop_6_gru_recurrent_kernel*
dtype0*
_output_shapes

: `f
strided_slice_9/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_9/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_9/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
strided_slice_9StridedSliceReadVariableOp_6:value:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
end_mask*
_output_shapes

:  *
Index0*
T0*

begin_maskn
MatMul_3MatMulzeros:output:0strided_slice_9:output:0*'
_output_shapes
:��������� *
T0�
ReadVariableOp_7ReadVariableOp%readvariableop_6_gru_recurrent_kernel^ReadVariableOp_6*
dtype0*
_output_shapes

: `g
strided_slice_10/stackConst*
valueB"        *
dtype0*
_output_shapes
:i
strided_slice_10/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:i
strided_slice_10/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_slice_10StridedSliceReadVariableOp_7:value:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
end_mask*
_output_shapes

:  *
T0*
Index0*

begin_masko
MatMul_4MatMulzeros:output:0strided_slice_10:output:0*
T0*'
_output_shapes
:��������� d
addAddV2BiasAdd:output:0MatMul_3:product:0*'
_output_shapes
:��������� *
T0J
ConstConst*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_1Const*
valueB
 *   ?*
dtype0*
_output_shapes
: U
MulMuladd:z:0Const:output:0*'
_output_shapes
:��������� *
T0Y
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:��������� \
clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*'
_output_shapes
:��������� *
T0T
clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:��������� h
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:��������� L
Const_2Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_3Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_2:output:0*'
_output_shapes
:��������� *
T0[
Add_3Add	Mul_1:z:0Const_3:output:0*'
_output_shapes
:��������� *
T0^
clip_by_value_1/Minimum/yConst*
dtype0*
_output_shapes
: *
valueB
 *  �?�
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:��������� V
clip_by_value_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*'
_output_shapes
:��������� *
T0c
mul_2Mulclip_by_value_1:z:0zeros:output:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_8ReadVariableOp%readvariableop_6_gru_recurrent_kernel^ReadVariableOp_7*
dtype0*
_output_shapes

: `g
strided_slice_11/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:i
strided_slice_11/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:i
strided_slice_11/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_11StridedSliceReadVariableOp_8:value:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
T0*
Index0j
MatMul_5MatMul	mul_2:z:0strided_slice_11:output:0*
T0*'
_output_shapes
:��������� h
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:��������� a
mul_3Mulclip_by_value:z:0zeros:output:0*
T0*'
_output_shapes
:��������� J
sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: _
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:��������� Q
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
valueB"����    *
dtype0�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
dtype0*
_output_shapes
: *
valueB :
���������T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0readvariableop_gru_kernelreadvariableop_3_gru_bias%readvariableop_6_gru_recurrent_kernel^ReadVariableOp_2^ReadVariableOp_5^ReadVariableOp_8*
condR
while_cond_4356*
_num_original_outputs
*
bodyR
while_body_4357*9
_output_shapes'
%: : : : :��������� : : : : : *
T
2
*8
output_shapes'
%: : : : :��������� : : : : : *
_lower_using_switch_merge(*
parallel_iterations K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
_output_shapes
: *
T0M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*'
_output_shapes
:��������� *
T0M
while/Identity_5Identitywhile:output:5*
_output_shapes
: *
T0M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
_output_shapes
: *
T0M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"����    *
dtype0*
_output_shapes
:�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :������������������ i
strided_slice_12/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:b
strided_slice_12/stack_1Const*
valueB: *
dtype0*
_output_shapes
:b
strided_slice_12/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_12StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:��������� *
Index0*
T0e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ �
IdentityIdentitystrided_slice_12:output:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8^while*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*?
_input_shapes.
,:������������������:::2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82
whilewhile2 
ReadVariableOpReadVariableOp: :( $
"
_user_specified_name
inputs/0: : 
�
�
"__inference_gru_layer_call_fn_4787
inputs_0&
"statefulpartitionedcall_gru_kernel$
 statefulpartitionedcall_gru_bias0
,statefulpartitionedcall_gru_recurrent_kernel
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0"statefulpartitionedcall_gru_kernel statefulpartitionedcall_gru_bias,statefulpartitionedcall_gru_recurrent_kernel*+
_gradient_op_typePartitionedCall-2391*F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_2390*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:��������� �
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*?
_input_shapes.
,:������������������:::22
StatefulPartitionedCallStatefulPartitionedCall: : : :( $
"
_user_specified_name
inputs/0
�
�
"__inference_gru_layer_call_fn_4231

inputs&
"statefulpartitionedcall_gru_kernel$
 statefulpartitionedcall_gru_bias0
,statefulpartitionedcall_gru_recurrent_kernel
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs"statefulpartitionedcall_gru_kernel statefulpartitionedcall_gru_bias,statefulpartitionedcall_gru_recurrent_kernel*F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_2692*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:��������� *+
_gradient_op_typePartitionedCall-2961�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*6
_input_shapes%
#:���������:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
�
�
while_cond_3808
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor

gru_kernel
gru_bias
gru_recurrent_kernel
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*>
_input_shapes-
+: : : : :��������� : : :::: : : : : : : :	 :  : 
�
�
?__inference_dense_layer_call_and_return_conditional_losses_2992

inputs&
"matmul_readvariableop_dense_kernel%
!biasadd_readvariableop_dense_bias
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpx
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel*
dtype0*
_output_shapes

: i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������t
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:���������*
T0P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:���������*
T0"
identityIdentity:output:0*.
_input_shapes
:��������� ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
�
D__inference_sequential_layer_call_and_return_conditional_losses_3044

inputs8
4features_statefulpartitionedcall_features_embeddings*
&gru_statefulpartitionedcall_gru_kernel(
$gru_statefulpartitionedcall_gru_bias4
0gru_statefulpartitionedcall_gru_recurrent_kernel.
*dense_statefulpartitionedcall_dense_kernel,
(dense_statefulpartitionedcall_dense_bias
identity��dense/StatefulPartitionedCall� features/StatefulPartitionedCall�gru/StatefulPartitionedCall�
 features/StatefulPartitionedCallStatefulPartitionedCallinputs4features_statefulpartitionedcall_features_embeddings*+
_output_shapes
:���������*
Tin
2*+
_gradient_op_typePartitionedCall-2417*K
fFRD
B__inference_features_layer_call_and_return_conditional_losses_2410*
Tout
2**
config_proto

GPU 

CPU2J 8�
gru/StatefulPartitionedCallStatefulPartitionedCall)features/StatefulPartitionedCall:output:0&gru_statefulpartitionedcall_gru_kernel$gru_statefulpartitionedcall_gru_bias0gru_statefulpartitionedcall_gru_recurrent_kernel*F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_2692*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:��������� *+
_gradient_op_typePartitionedCall-2961�
dense/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0*dense_statefulpartitionedcall_dense_kernel(dense_statefulpartitionedcall_dense_bias*'
_output_shapes
:���������*
Tin
2*+
_gradient_op_typePartitionedCall-2999*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_2992*
Tout
2**
config_proto

GPU 

CPU2J 8�
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall!^features/StatefulPartitionedCall^gru/StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*>
_input_shapes-
+:���������::::::2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2D
 features/StatefulPartitionedCall features/StatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
��
�
__inference__wrapped_model_1718
features_input<
8sequential_features_embedding_lookup_features_embeddings,
(sequential_gru_readvariableop_gru_kernel,
(sequential_gru_readvariableop_3_gru_bias8
4sequential_gru_readvariableop_6_gru_recurrent_kernel7
3sequential_dense_matmul_readvariableop_dense_kernel6
2sequential_dense_biasadd_readvariableop_dense_bias
identity��'sequential/dense/BiasAdd/ReadVariableOp�&sequential/dense/MatMul/ReadVariableOp�$sequential/features/embedding_lookup�sequential/gru/ReadVariableOp�sequential/gru/ReadVariableOp_1�sequential/gru/ReadVariableOp_2�sequential/gru/ReadVariableOp_3�sequential/gru/ReadVariableOp_4�sequential/gru/ReadVariableOp_5�sequential/gru/ReadVariableOp_6�sequential/gru/ReadVariableOp_7�sequential/gru/ReadVariableOp_8�sequential/gru/whileq
sequential/features/CastCastfeatures_input*

DstT0*'
_output_shapes
:���������*

SrcT0�
$sequential/features/embedding_lookupResourceGather8sequential_features_embedding_lookup_features_embeddingssequential/features/Cast:y:0*K
_classA
?=loc:@sequential/features/embedding_lookup/features/embeddings*
Tindices0*
dtype0*+
_output_shapes
:����������
-sequential/features/embedding_lookup/IdentityIdentity-sequential/features/embedding_lookup:output:0*
T0*K
_classA
?=loc:@sequential/features/embedding_lookup/features/embeddings*+
_output_shapes
:����������
/sequential/features/embedding_lookup/Identity_1Identity6sequential/features/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������|
sequential/gru/ShapeShape8sequential/features/embedding_lookup/Identity_1:output:0*
_output_shapes
:*
T0l
"sequential/gru/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:n
$sequential/gru/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:n
$sequential/gru/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
sequential/gru/strided_sliceStridedSlicesequential/gru/Shape:output:0+sequential/gru/strided_slice/stack:output:0-sequential/gru/strided_slice/stack_1:output:0-sequential/gru/strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0\
sequential/gru/zeros/mul/yConst*
value	B : *
dtype0*
_output_shapes
: �
sequential/gru/zeros/mulMul%sequential/gru/strided_slice:output:0#sequential/gru/zeros/mul/y:output:0*
T0*
_output_shapes
: ^
sequential/gru/zeros/Less/yConst*
dtype0*
_output_shapes
: *
value
B :��
sequential/gru/zeros/LessLesssequential/gru/zeros/mul:z:0$sequential/gru/zeros/Less/y:output:0*
_output_shapes
: *
T0_
sequential/gru/zeros/packed/1Const*
value	B : *
dtype0*
_output_shapes
: �
sequential/gru/zeros/packedPack%sequential/gru/strided_slice:output:0&sequential/gru/zeros/packed/1:output:0*
N*
_output_shapes
:*
T0_
sequential/gru/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: �
sequential/gru/zerosFill$sequential/gru/zeros/packed:output:0#sequential/gru/zeros/Const:output:0*
T0*'
_output_shapes
:��������� r
sequential/gru/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
sequential/gru/transpose	Transpose8sequential/features/embedding_lookup/Identity_1:output:0&sequential/gru/transpose/perm:output:0*+
_output_shapes
:���������*
T0b
sequential/gru/Shape_1Shapesequential/gru/transpose:y:0*
_output_shapes
:*
T0n
$sequential/gru/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:p
&sequential/gru/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:p
&sequential/gru/strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
sequential/gru/strided_slice_1StridedSlicesequential/gru/Shape_1:output:0-sequential/gru/strided_slice_1/stack:output:0/sequential/gru/strided_slice_1/stack_1:output:0/sequential/gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: u
*sequential/gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
valueB :
���������*
dtype0�
sequential/gru/TensorArrayV2TensorListReserve3sequential/gru/TensorArrayV2/element_shape:output:0'sequential/gru/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: �
Dsequential/gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
6sequential/gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsequential/gru/transpose:y:0Msequential/gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: n
$sequential/gru/strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:p
&sequential/gru/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:p
&sequential/gru/strided_slice_2/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
sequential/gru/strided_slice_2StridedSlicesequential/gru/transpose:y:0-sequential/gru/strided_slice_2/stack:output:0/sequential/gru/strided_slice_2/stack_1:output:0/sequential/gru/strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:���������*
T0*
Index0�
sequential/gru/ReadVariableOpReadVariableOp(sequential_gru_readvariableop_gru_kernel*
dtype0*
_output_shapes

:`u
$sequential/gru/strided_slice_3/stackConst*
valueB"        *
dtype0*
_output_shapes
:w
&sequential/gru/strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB"        w
&sequential/gru/strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
sequential/gru/strided_slice_3StridedSlice%sequential/gru/ReadVariableOp:value:0-sequential/gru/strided_slice_3/stack:output:0/sequential/gru/strided_slice_3/stack_1:output:0/sequential/gru/strided_slice_3/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
T0*
Index0�
sequential/gru/MatMulMatMul'sequential/gru/strided_slice_2:output:0'sequential/gru/strided_slice_3:output:0*
T0*'
_output_shapes
:��������� �
sequential/gru/ReadVariableOp_1ReadVariableOp(sequential_gru_readvariableop_gru_kernel^sequential/gru/ReadVariableOp*
dtype0*
_output_shapes

:`u
$sequential/gru/strided_slice_4/stackConst*
valueB"        *
dtype0*
_output_shapes
:w
&sequential/gru/strided_slice_4/stack_1Const*
dtype0*
_output_shapes
:*
valueB"    @   w
&sequential/gru/strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
sequential/gru/strided_slice_4StridedSlice'sequential/gru/ReadVariableOp_1:value:0-sequential/gru/strided_slice_4/stack:output:0/sequential/gru/strided_slice_4/stack_1:output:0/sequential/gru/strided_slice_4/stack_2:output:0*
end_mask*
_output_shapes

: *
Index0*
T0*

begin_mask�
sequential/gru/MatMul_1MatMul'sequential/gru/strided_slice_2:output:0'sequential/gru/strided_slice_4:output:0*
T0*'
_output_shapes
:��������� �
sequential/gru/ReadVariableOp_2ReadVariableOp(sequential_gru_readvariableop_gru_kernel ^sequential/gru/ReadVariableOp_1*
dtype0*
_output_shapes

:`u
$sequential/gru/strided_slice_5/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:w
&sequential/gru/strided_slice_5/stack_1Const*
_output_shapes
:*
valueB"        *
dtype0w
&sequential/gru/strided_slice_5/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
sequential/gru/strided_slice_5StridedSlice'sequential/gru/ReadVariableOp_2:value:0-sequential/gru/strided_slice_5/stack:output:0/sequential/gru/strided_slice_5/stack_1:output:0/sequential/gru/strided_slice_5/stack_2:output:0*
end_mask*
_output_shapes

: *
Index0*
T0*

begin_mask�
sequential/gru/MatMul_2MatMul'sequential/gru/strided_slice_2:output:0'sequential/gru/strided_slice_5:output:0*
T0*'
_output_shapes
:��������� �
sequential/gru/ReadVariableOp_3ReadVariableOp(sequential_gru_readvariableop_3_gru_bias*
dtype0*
_output_shapes
:`n
$sequential/gru/strided_slice_6/stackConst*
valueB: *
dtype0*
_output_shapes
:p
&sequential/gru/strided_slice_6/stack_1Const*
valueB: *
dtype0*
_output_shapes
:p
&sequential/gru/strided_slice_6/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
sequential/gru/strided_slice_6StridedSlice'sequential/gru/ReadVariableOp_3:value:0-sequential/gru/strided_slice_6/stack:output:0/sequential/gru/strided_slice_6/stack_1:output:0/sequential/gru/strided_slice_6/stack_2:output:0*
_output_shapes
: *
T0*
Index0*

begin_mask�
sequential/gru/BiasAddBiasAddsequential/gru/MatMul:product:0'sequential/gru/strided_slice_6:output:0*'
_output_shapes
:��������� *
T0�
sequential/gru/ReadVariableOp_4ReadVariableOp(sequential_gru_readvariableop_3_gru_bias ^sequential/gru/ReadVariableOp_3*
dtype0*
_output_shapes
:`n
$sequential/gru/strided_slice_7/stackConst*
valueB: *
dtype0*
_output_shapes
:p
&sequential/gru/strided_slice_7/stack_1Const*
dtype0*
_output_shapes
:*
valueB:@p
&sequential/gru/strided_slice_7/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
sequential/gru/strided_slice_7StridedSlice'sequential/gru/ReadVariableOp_4:value:0-sequential/gru/strided_slice_7/stack:output:0/sequential/gru/strided_slice_7/stack_1:output:0/sequential/gru/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: �
sequential/gru/BiasAdd_1BiasAdd!sequential/gru/MatMul_1:product:0'sequential/gru/strided_slice_7:output:0*
T0*'
_output_shapes
:��������� �
sequential/gru/ReadVariableOp_5ReadVariableOp(sequential_gru_readvariableop_3_gru_bias ^sequential/gru/ReadVariableOp_4*
dtype0*
_output_shapes
:`n
$sequential/gru/strided_slice_8/stackConst*
dtype0*
_output_shapes
:*
valueB:@p
&sequential/gru/strided_slice_8/stack_1Const*
_output_shapes
:*
valueB: *
dtype0p
&sequential/gru/strided_slice_8/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
sequential/gru/strided_slice_8StridedSlice'sequential/gru/ReadVariableOp_5:value:0-sequential/gru/strided_slice_8/stack:output:0/sequential/gru/strided_slice_8/stack_1:output:0/sequential/gru/strided_slice_8/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
end_mask�
sequential/gru/BiasAdd_2BiasAdd!sequential/gru/MatMul_2:product:0'sequential/gru/strided_slice_8:output:0*
T0*'
_output_shapes
:��������� �
sequential/gru/ReadVariableOp_6ReadVariableOp4sequential_gru_readvariableop_6_gru_recurrent_kernel*
dtype0*
_output_shapes

: `u
$sequential/gru/strided_slice_9/stackConst*
_output_shapes
:*
valueB"        *
dtype0w
&sequential/gru/strided_slice_9/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:w
&sequential/gru/strided_slice_9/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
sequential/gru/strided_slice_9StridedSlice'sequential/gru/ReadVariableOp_6:value:0-sequential/gru/strided_slice_9/stack:output:0/sequential/gru/strided_slice_9/stack_1:output:0/sequential/gru/strided_slice_9/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:  �
sequential/gru/MatMul_3MatMulsequential/gru/zeros:output:0'sequential/gru/strided_slice_9:output:0*
T0*'
_output_shapes
:��������� �
sequential/gru/ReadVariableOp_7ReadVariableOp4sequential_gru_readvariableop_6_gru_recurrent_kernel ^sequential/gru/ReadVariableOp_6*
dtype0*
_output_shapes

: `v
%sequential/gru/strided_slice_10/stackConst*
_output_shapes
:*
valueB"        *
dtype0x
'sequential/gru/strided_slice_10/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:x
'sequential/gru/strided_slice_10/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
sequential/gru/strided_slice_10StridedSlice'sequential/gru/ReadVariableOp_7:value:0.sequential/gru/strided_slice_10/stack:output:00sequential/gru/strided_slice_10/stack_1:output:00sequential/gru/strided_slice_10/stack_2:output:0*
end_mask*
_output_shapes

:  *
Index0*
T0*

begin_mask�
sequential/gru/MatMul_4MatMulsequential/gru/zeros:output:0(sequential/gru/strided_slice_10:output:0*'
_output_shapes
:��������� *
T0�
sequential/gru/addAddV2sequential/gru/BiasAdd:output:0!sequential/gru/MatMul_3:product:0*
T0*'
_output_shapes
:��������� Y
sequential/gru/ConstConst*
valueB
 *��L>*
dtype0*
_output_shapes
: [
sequential/gru/Const_1Const*
valueB
 *   ?*
dtype0*
_output_shapes
: �
sequential/gru/MulMulsequential/gru/add:z:0sequential/gru/Const:output:0*
T0*'
_output_shapes
:��������� �
sequential/gru/Add_1Addsequential/gru/Mul:z:0sequential/gru/Const_1:output:0*
T0*'
_output_shapes
:��������� k
&sequential/gru/clip_by_value/Minimum/yConst*
_output_shapes
: *
valueB
 *  �?*
dtype0�
$sequential/gru/clip_by_value/MinimumMinimumsequential/gru/Add_1:z:0/sequential/gru/clip_by_value/Minimum/y:output:0*'
_output_shapes
:��������� *
T0c
sequential/gru/clip_by_value/yConst*
_output_shapes
: *
valueB
 *    *
dtype0�
sequential/gru/clip_by_valueMaximum(sequential/gru/clip_by_value/Minimum:z:0'sequential/gru/clip_by_value/y:output:0*
T0*'
_output_shapes
:��������� �
sequential/gru/add_2AddV2!sequential/gru/BiasAdd_1:output:0!sequential/gru/MatMul_4:product:0*'
_output_shapes
:��������� *
T0[
sequential/gru/Const_2Const*
valueB
 *��L>*
dtype0*
_output_shapes
: [
sequential/gru/Const_3Const*
valueB
 *   ?*
dtype0*
_output_shapes
: �
sequential/gru/Mul_1Mulsequential/gru/add_2:z:0sequential/gru/Const_2:output:0*'
_output_shapes
:��������� *
T0�
sequential/gru/Add_3Addsequential/gru/Mul_1:z:0sequential/gru/Const_3:output:0*
T0*'
_output_shapes
:��������� m
(sequential/gru/clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
&sequential/gru/clip_by_value_1/MinimumMinimumsequential/gru/Add_3:z:01sequential/gru/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:��������� e
 sequential/gru/clip_by_value_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *    �
sequential/gru/clip_by_value_1Maximum*sequential/gru/clip_by_value_1/Minimum:z:0)sequential/gru/clip_by_value_1/y:output:0*'
_output_shapes
:��������� *
T0�
sequential/gru/mul_2Mul"sequential/gru/clip_by_value_1:z:0sequential/gru/zeros:output:0*
T0*'
_output_shapes
:��������� �
sequential/gru/ReadVariableOp_8ReadVariableOp4sequential_gru_readvariableop_6_gru_recurrent_kernel ^sequential/gru/ReadVariableOp_7*
dtype0*
_output_shapes

: `v
%sequential/gru/strided_slice_11/stackConst*
dtype0*
_output_shapes
:*
valueB"    @   x
'sequential/gru/strided_slice_11/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:x
'sequential/gru/strided_slice_11/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
sequential/gru/strided_slice_11StridedSlice'sequential/gru/ReadVariableOp_8:value:0.sequential/gru/strided_slice_11/stack:output:00sequential/gru/strided_slice_11/stack_1:output:00sequential/gru/strided_slice_11/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
T0*
Index0�
sequential/gru/MatMul_5MatMulsequential/gru/mul_2:z:0(sequential/gru/strided_slice_11:output:0*'
_output_shapes
:��������� *
T0�
sequential/gru/add_4AddV2!sequential/gru/BiasAdd_2:output:0!sequential/gru/MatMul_5:product:0*
T0*'
_output_shapes
:��������� g
sequential/gru/TanhTanhsequential/gru/add_4:z:0*
T0*'
_output_shapes
:��������� �
sequential/gru/mul_3Mul sequential/gru/clip_by_value:z:0sequential/gru/zeros:output:0*'
_output_shapes
:��������� *
T0Y
sequential/gru/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
sequential/gru/subSubsequential/gru/sub/x:output:0 sequential/gru/clip_by_value:z:0*
T0*'
_output_shapes
:��������� ~
sequential/gru/mul_4Mulsequential/gru/sub:z:0sequential/gru/Tanh:y:0*'
_output_shapes
:��������� *
T0�
sequential/gru/add_5AddV2sequential/gru/mul_3:z:0sequential/gru/mul_4:z:0*
T0*'
_output_shapes
:��������� }
,sequential/gru/TensorArrayV2_1/element_shapeConst*
valueB"����    *
dtype0*
_output_shapes
:�
sequential/gru/TensorArrayV2_1TensorListReserve5sequential/gru/TensorArrayV2_1/element_shape:output:0'sequential/gru/strided_slice_1:output:0*
element_dtype0*
_output_shapes
: *

shape_type0U
sequential/gru/timeConst*
value	B : *
dtype0*
_output_shapes
: r
'sequential/gru/while/maximum_iterationsConst*
dtype0*
_output_shapes
: *
valueB :
���������c
!sequential/gru/while/loop_counterConst*
dtype0*
_output_shapes
: *
value	B : �
sequential/gru/whileWhile*sequential/gru/while/loop_counter:output:00sequential/gru/while/maximum_iterations:output:0sequential/gru/time:output:0'sequential/gru/TensorArrayV2_1:handle:0sequential/gru/zeros:output:0'sequential/gru/strided_slice_1:output:0Fsequential/gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0(sequential_gru_readvariableop_gru_kernel(sequential_gru_readvariableop_3_gru_bias4sequential_gru_readvariableop_6_gru_recurrent_kernel ^sequential/gru/ReadVariableOp_2 ^sequential/gru/ReadVariableOp_5 ^sequential/gru/ReadVariableOp_8**
body"R 
sequential_gru_while_body_1563*9
_output_shapes'
%: : : : :��������� : : : : : *
T
2
*8
output_shapes'
%: : : : :��������� : : : : : *
_lower_using_switch_merge(*
parallel_iterations **
cond"R 
sequential_gru_while_cond_1562*
_num_original_outputs
i
sequential/gru/while/IdentityIdentitysequential/gru/while:output:0*
T0*
_output_shapes
: k
sequential/gru/while/Identity_1Identitysequential/gru/while:output:1*
T0*
_output_shapes
: k
sequential/gru/while/Identity_2Identitysequential/gru/while:output:2*
T0*
_output_shapes
: k
sequential/gru/while/Identity_3Identitysequential/gru/while:output:3*
T0*
_output_shapes
: |
sequential/gru/while/Identity_4Identitysequential/gru/while:output:4*
T0*'
_output_shapes
:��������� k
sequential/gru/while/Identity_5Identitysequential/gru/while:output:5*
T0*
_output_shapes
: k
sequential/gru/while/Identity_6Identitysequential/gru/while:output:6*
T0*
_output_shapes
: k
sequential/gru/while/Identity_7Identitysequential/gru/while:output:7*
T0*
_output_shapes
: k
sequential/gru/while/Identity_8Identitysequential/gru/while:output:8*
T0*
_output_shapes
: k
sequential/gru/while/Identity_9Identitysequential/gru/while:output:9*
T0*
_output_shapes
: �
?sequential/gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"����    *
dtype0*
_output_shapes
:�
1sequential/gru/TensorArrayV2Stack/TensorListStackTensorListStack(sequential/gru/while/Identity_3:output:0Hsequential/gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:��������� x
%sequential/gru/strided_slice_12/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:q
'sequential/gru/strided_slice_12/stack_1Const*
dtype0*
_output_shapes
:*
valueB: q
'sequential/gru/strided_slice_12/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
sequential/gru/strided_slice_12StridedSlice:sequential/gru/TensorArrayV2Stack/TensorListStack:tensor:0.sequential/gru/strided_slice_12/stack:output:00sequential/gru/strided_slice_12/stack_1:output:00sequential/gru/strided_slice_12/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*'
_output_shapes
:��������� t
sequential/gru/transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
sequential/gru/transpose_1	Transpose:sequential/gru/TensorArrayV2Stack/TensorListStack:tensor:0(sequential/gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� �
&sequential/dense/MatMul/ReadVariableOpReadVariableOp3sequential_dense_matmul_readvariableop_dense_kernel*
dtype0*
_output_shapes

: �
sequential/dense/MatMulMatMul(sequential/gru/strided_slice_12:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_biasadd_readvariableop_dense_bias*
dtype0*
_output_shapes
:�
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentity#sequential/dense/Relu:activations:0(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp%^sequential/features/embedding_lookup^sequential/gru/ReadVariableOp ^sequential/gru/ReadVariableOp_1 ^sequential/gru/ReadVariableOp_2 ^sequential/gru/ReadVariableOp_3 ^sequential/gru/ReadVariableOp_4 ^sequential/gru/ReadVariableOp_5 ^sequential/gru/ReadVariableOp_6 ^sequential/gru/ReadVariableOp_7 ^sequential/gru/ReadVariableOp_8^sequential/gru/while*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*>
_input_shapes-
+:���������::::::2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2>
sequential/gru/ReadVariableOpsequential/gru/ReadVariableOp2B
sequential/gru/ReadVariableOp_1sequential/gru/ReadVariableOp_12B
sequential/gru/ReadVariableOp_2sequential/gru/ReadVariableOp_22B
sequential/gru/ReadVariableOp_3sequential/gru/ReadVariableOp_32B
sequential/gru/ReadVariableOp_4sequential/gru/ReadVariableOp_42B
sequential/gru/ReadVariableOp_5sequential/gru/ReadVariableOp_52,
sequential/gru/whilesequential/gru/while2B
sequential/gru/ReadVariableOp_6sequential/gru/ReadVariableOp_62B
sequential/gru/ReadVariableOp_7sequential/gru/ReadVariableOp_72B
sequential/gru/ReadVariableOp_8sequential/gru/ReadVariableOp_82R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2L
$sequential/features/embedding_lookup$sequential/features/embedding_lookup: : : : :. *
(
_user_specified_namefeatures_input: : 
�j
�
while_body_2544
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_gru_kernel_0
readvariableop_3_gru_bias_0+
'readvariableop_6_gru_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
readvariableop_gru_kernel
readvariableop_3_gru_bias)
%readvariableop_6_gru_recurrent_kernel��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�ReadVariableOp_7�ReadVariableOp_8�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:���������j
ReadVariableOpReadVariableOpreadvariableop_gru_kernel_0*
_output_shapes

:`*
dtype0d
strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

: �
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice:output:0*'
_output_shapes
:��������� *
T0}
ReadVariableOp_1ReadVariableOpreadvariableop_gru_kernel_0^ReadVariableOp*
dtype0*
_output_shapes

:`f
strided_slice_2/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_2/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_2StridedSliceReadVariableOp_1:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
end_mask*
_output_shapes

: *
Index0*
T0*

begin_mask�
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_2:output:0*
T0*'
_output_shapes
:��������� 
ReadVariableOp_2ReadVariableOpreadvariableop_gru_kernel_0^ReadVariableOp_1*
dtype0*
_output_shapes

:`f
strided_slice_3/stackConst*
_output_shapes
:*
valueB"    @   *
dtype0h
strided_slice_3/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_3/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_3StridedSliceReadVariableOp_2:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

: �
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_3:output:0*'
_output_shapes
:��������� *
T0h
ReadVariableOp_3ReadVariableOpreadvariableop_3_gru_bias_0*
dtype0*
_output_shapes
:`_
strided_slice_4/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_4/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_4/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_4StridedSliceReadVariableOp_3:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*

begin_mask*
_output_shapes
: *
T0*
Index0p
BiasAddBiasAddMatMul:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:��������� {
ReadVariableOp_4ReadVariableOpreadvariableop_3_gru_bias_0^ReadVariableOp_3*
dtype0*
_output_shapes
:`_
strided_slice_5/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_5/stack_1Const*
valueB:@*
dtype0*
_output_shapes
:a
strided_slice_5/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_5StridedSliceReadVariableOp_4:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
T0*
Index0*
_output_shapes
: t
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:��������� {
ReadVariableOp_5ReadVariableOpreadvariableop_3_gru_bias_0^ReadVariableOp_4*
dtype0*
_output_shapes
:`_
strided_slice_6/stackConst*
dtype0*
_output_shapes
:*
valueB:@a
strided_slice_6/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_6/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_6StridedSliceReadVariableOp_5:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
T0*
Index0*
end_mask*
_output_shapes
: t
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_6:output:0*'
_output_shapes
:��������� *
T0x
ReadVariableOp_6ReadVariableOp'readvariableop_6_gru_recurrent_kernel_0*
dtype0*
_output_shapes

: `f
strided_slice_7/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_7/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_7/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_slice_7StridedSliceReadVariableOp_6:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:  m
MatMul_3MatMulplaceholder_2strided_slice_7:output:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_7ReadVariableOp'readvariableop_6_gru_recurrent_kernel_0^ReadVariableOp_6*
dtype0*
_output_shapes

: `f
strided_slice_8/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_8/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_8/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
strided_slice_8StridedSliceReadVariableOp_7:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
Index0*
T0m
MatMul_4MatMulplaceholder_2strided_slice_8:output:0*
T0*'
_output_shapes
:��������� d
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:��������� J
ConstConst*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_1Const*
valueB
 *   ?*
dtype0*
_output_shapes
: U
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:��������� Y
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:��������� \
clip_by_value/Minimum/yConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*'
_output_shapes
:��������� *
T0T
clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*'
_output_shapes
:��������� *
T0h
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*'
_output_shapes
:��������� *
T0L
Const_2Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_3Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:��������� [
Add_3Add	Mul_1:z:0Const_3:output:0*'
_output_shapes
:��������� *
T0^
clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:��������� V
clip_by_value_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:��������� b
mul_2Mulclip_by_value_1:z:0placeholder_2*
T0*'
_output_shapes
:��������� �
ReadVariableOp_8ReadVariableOp'readvariableop_6_gru_recurrent_kernel_0^ReadVariableOp_7*
dtype0*
_output_shapes

: `f
strided_slice_9/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_9/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_9/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_slice_9StridedSliceReadVariableOp_8:value:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
T0*
Index0i
MatMul_5MatMul	mul_2:z:0strided_slice_9:output:0*
T0*'
_output_shapes
:��������� h
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:��������� `
mul_3Mulclip_by_value:z:0placeholder_2*'
_output_shapes
:��������� *
T0J
sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: _
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:��������� Q
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:��������� �
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_5:z:0*
element_dtype0*
_output_shapes
: I
add_6/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_6AddV2placeholderadd_6/y:output:0*
_output_shapes
: *
T0I
add_7/yConst*
dtype0*
_output_shapes
: *
value	B :U
add_7AddV2while_loop_counteradd_7/y:output:0*
_output_shapes
: *
T0�
IdentityIdentity	add_7:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
_output_shapes
: *
T0�

Identity_1Identitywhile_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
_output_shapes
: *
T0�

Identity_2Identity	add_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
_output_shapes
: *
T0�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
_output_shapes
: *
T0�

Identity_4Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*'
_output_shapes
:��������� *
T0"$
strided_slice_1strided_slice_1_0"P
%readvariableop_6_gru_recurrent_kernel'readvariableop_6_gru_recurrent_kernel_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"8
readvariableop_3_gru_biasreadvariableop_3_gru_bias_0"8
readvariableop_gru_kernelreadvariableop_gru_kernel_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*>
_input_shapes-
+: : : : :��������� : : :::2$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_2: : : :	 :  : : : : : 
�U
�
B__inference_gru_cell_layer_call_and_return_conditional_losses_4894

inputs
states_0
readvariableop_gru_kernel
readvariableop_3_gru_bias)
%readvariableop_6_gru_recurrent_kernel
identity

identity_1��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�ReadVariableOp_7�ReadVariableOp_8h
ReadVariableOpReadVariableOpreadvariableop_gru_kernel*
dtype0*
_output_shapes

:`d
strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
Index0*
T0b
MatMulMatMulinputsstrided_slice:output:0*'
_output_shapes
:��������� *
T0{
ReadVariableOp_1ReadVariableOpreadvariableop_gru_kernel^ReadVariableOp*
_output_shapes

:`*
dtype0f
strided_slice_1/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_1/stack_1Const*
_output_shapes
:*
valueB"    @   *
dtype0h
strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
end_mask*
_output_shapes

: *
T0*
Index0*

begin_maskf
MatMul_1MatMulinputsstrided_slice_1:output:0*
T0*'
_output_shapes
:��������� }
ReadVariableOp_2ReadVariableOpreadvariableop_gru_kernel^ReadVariableOp_1*
dtype0*
_output_shapes

:`f
strided_slice_2/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_2/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
end_mask*
_output_shapes

: *
T0*
Index0*

begin_maskf
MatMul_2MatMulinputsstrided_slice_2:output:0*'
_output_shapes
:��������� *
T0f
ReadVariableOp_3ReadVariableOpreadvariableop_3_gru_bias*
dtype0*
_output_shapes
:`_
strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*

begin_mask*
_output_shapes
: *
Index0*
T0p
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*'
_output_shapes
:��������� y
ReadVariableOp_4ReadVariableOpreadvariableop_3_gru_bias^ReadVariableOp_3*
dtype0*
_output_shapes
:`_
strided_slice_4/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_4/stack_1Const*
valueB:@*
dtype0*
_output_shapes
:a
strided_slice_4/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
strided_slice_4StridedSliceReadVariableOp_4:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
_output_shapes
: *
T0*
Index0t
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:��������� y
ReadVariableOp_5ReadVariableOpreadvariableop_3_gru_bias^ReadVariableOp_4*
dtype0*
_output_shapes
:`_
strided_slice_5/stackConst*
valueB:@*
dtype0*
_output_shapes
:a
strided_slice_5/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_5/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_5StridedSliceReadVariableOp_5:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
end_mask*
_output_shapes
: *
T0*
Index0t
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:��������� v
ReadVariableOp_6ReadVariableOp%readvariableop_6_gru_recurrent_kernel*
dtype0*
_output_shapes

: `f
strided_slice_6/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_6/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_6/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_6StridedSliceReadVariableOp_6:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:  h
MatMul_3MatMulstates_0strided_slice_6:output:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_7ReadVariableOp%readvariableop_6_gru_recurrent_kernel^ReadVariableOp_6*
_output_shapes

: `*
dtype0f
strided_slice_7/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_7/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_7/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_slice_7StridedSliceReadVariableOp_7:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:  h
MatMul_4MatMulstates_0strided_slice_7:output:0*'
_output_shapes
:��������� *
T0d
addAddV2BiasAdd:output:0MatMul_3:product:0*'
_output_shapes
:��������� *
T0J
ConstConst*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_1Const*
valueB
 *   ?*
dtype0*
_output_shapes
: U
MulMuladd:z:0Const:output:0*'
_output_shapes
:��������� *
T0Y
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:��������� \
clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:��������� T
clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:��������� h
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:��������� L
Const_2Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_3Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:��������� [
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:��������� ^
clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*'
_output_shapes
:��������� *
T0V
clip_by_value_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:��������� ]
mul_2Mulclip_by_value_1:z:0states_0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_8ReadVariableOp%readvariableop_6_gru_recurrent_kernel^ReadVariableOp_7*
dtype0*
_output_shapes

: `f
strided_slice_8/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_8/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_8/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_8StridedSliceReadVariableOp_8:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
_output_shapes

:  *
Index0*
T0*

begin_mask*
end_maski
MatMul_5MatMul	mul_2:z:0strided_slice_8:output:0*'
_output_shapes
:��������� *
T0h
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:��������� [
mul_3Mulclip_by_value:z:0states_0*'
_output_shapes
:��������� *
T0J
sub/xConst*
_output_shapes
: *
valueB
 *  �?*
dtype0_
subSubsub/x:output:0clip_by_value:z:0*'
_output_shapes
:��������� *
T0Q
mul_4Mulsub:z:0Tanh:y:0*'
_output_shapes
:��������� *
T0V
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:��������� �
IdentityIdentity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*'
_output_shapes
:��������� *
T0�

Identity_1Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0"
identityIdentity:output:0*E
_input_shapes4
2:���������:��������� :::2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82 
ReadVariableOpReadVariableOp: :& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0: : 
�
�
while_cond_4622
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor

gru_kernel
gru_bias
gru_recurrent_kernel
identity
P
LessLessplaceholderless_strided_slice_1*
_output_shapes
: *
T0?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*>
_input_shapes-
+: : : : :��������� : : ::::  : : : : : : : : :	 
��
�
=__inference_gru_layer_call_and_return_conditional_losses_3957

inputs
readvariableop_gru_kernel
readvariableop_3_gru_bias)
%readvariableop_6_gru_recurrent_kernel
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�ReadVariableOp_7�ReadVariableOp_8�while;
ShapeShapeinputs*
_output_shapes
:*
T0]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
_output_shapes
: *
T0*
Index0*
shrink_axis_maskM
zeros/mul/yConst*
_output_shapes
: *
value	B : *
dtype0_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
value
B :�*
dtype0Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B : *
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*'
_output_shapes
:��������� *
T0c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
_output_shapes
: *
T0*
Index0*
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
dtype0*
_output_shapes
: *
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: �
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:���������*
Index0*
T0h
ReadVariableOpReadVariableOpreadvariableop_gru_kernel*
dtype0*
_output_shapes

:`f
strided_slice_3/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_3/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
strided_slice_3StridedSliceReadVariableOp:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

: v
MatMulMatMulstrided_slice_2:output:0strided_slice_3:output:0*
T0*'
_output_shapes
:��������� {
ReadVariableOp_1ReadVariableOpreadvariableop_gru_kernel^ReadVariableOp*
dtype0*
_output_shapes

:`f
strided_slice_4/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_4/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_4/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_slice_4StridedSliceReadVariableOp_1:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
end_mask*
_output_shapes

: *
Index0*
T0*

begin_maskx
MatMul_1MatMulstrided_slice_2:output:0strided_slice_4:output:0*
T0*'
_output_shapes
:��������� }
ReadVariableOp_2ReadVariableOpreadvariableop_gru_kernel^ReadVariableOp_1*
dtype0*
_output_shapes

:`f
strided_slice_5/stackConst*
_output_shapes
:*
valueB"    @   *
dtype0h
strided_slice_5/stack_1Const*
_output_shapes
:*
valueB"        *
dtype0h
strided_slice_5/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_slice_5StridedSliceReadVariableOp_2:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
Index0*
T0x
MatMul_2MatMulstrided_slice_2:output:0strided_slice_5:output:0*
T0*'
_output_shapes
:��������� f
ReadVariableOp_3ReadVariableOpreadvariableop_3_gru_bias*
dtype0*
_output_shapes
:`_
strided_slice_6/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_6/stack_1Const*
_output_shapes
:*
valueB: *
dtype0a
strided_slice_6/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
strided_slice_6StridedSliceReadVariableOp_3:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*

begin_mask*
_output_shapes
: *
Index0*
T0p
BiasAddBiasAddMatMul:product:0strided_slice_6:output:0*'
_output_shapes
:��������� *
T0y
ReadVariableOp_4ReadVariableOpreadvariableop_3_gru_bias^ReadVariableOp_3*
dtype0*
_output_shapes
:`_
strided_slice_7/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_7/stack_1Const*
dtype0*
_output_shapes
:*
valueB:@a
strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_7StridedSliceReadVariableOp_4:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
_output_shapes
: *
Index0*
T0t
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_7:output:0*
T0*'
_output_shapes
:��������� y
ReadVariableOp_5ReadVariableOpreadvariableop_3_gru_bias^ReadVariableOp_4*
dtype0*
_output_shapes
:`_
strided_slice_8/stackConst*
dtype0*
_output_shapes
:*
valueB:@a
strided_slice_8/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_8/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_8StridedSliceReadVariableOp_5:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
end_mask*
_output_shapes
: *
Index0*
T0t
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_8:output:0*
T0*'
_output_shapes
:��������� v
ReadVariableOp_6ReadVariableOp%readvariableop_6_gru_recurrent_kernel*
_output_shapes

: `*
dtype0f
strided_slice_9/stackConst*
dtype0*
_output_shapes
:*
valueB"        h
strided_slice_9/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_9/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_9StridedSliceReadVariableOp_6:value:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
T0*
Index0n
MatMul_3MatMulzeros:output:0strided_slice_9:output:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_7ReadVariableOp%readvariableop_6_gru_recurrent_kernel^ReadVariableOp_6*
dtype0*
_output_shapes

: `g
strided_slice_10/stackConst*
valueB"        *
dtype0*
_output_shapes
:i
strided_slice_10/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:i
strided_slice_10/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_10StridedSliceReadVariableOp_7:value:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
T0*
Index0o
MatMul_4MatMulzeros:output:0strided_slice_10:output:0*
T0*'
_output_shapes
:��������� d
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:��������� J
ConstConst*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *   ?U
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:��������� Y
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:��������� \
clip_by_value/Minimum/yConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:��������� T
clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:��������� h
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*'
_output_shapes
:��������� *
T0L
Const_2Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_3Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_2:output:0*'
_output_shapes
:��������� *
T0[
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:��������� ^
clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:��������� V
clip_by_value_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:��������� c
mul_2Mulclip_by_value_1:z:0zeros:output:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_8ReadVariableOp%readvariableop_6_gru_recurrent_kernel^ReadVariableOp_7*
dtype0*
_output_shapes

: `g
strided_slice_11/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:i
strided_slice_11/stack_1Const*
dtype0*
_output_shapes
:*
valueB"        i
strided_slice_11/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_11StridedSliceReadVariableOp_8:value:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:  j
MatMul_5MatMul	mul_2:z:0strided_slice_11:output:0*
T0*'
_output_shapes
:��������� h
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:��������� a
mul_3Mulclip_by_value:z:0zeros:output:0*'
_output_shapes
:��������� *
T0J
sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?_
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:��������� Q
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
valueB"����    *
dtype0*
_output_shapes
:�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
valueB :
���������*
dtype0*
_output_shapes
: T
while/loop_counterConst*
_output_shapes
: *
value	B : *
dtype0�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0readvariableop_gru_kernelreadvariableop_3_gru_bias%readvariableop_6_gru_recurrent_kernel^ReadVariableOp_2^ReadVariableOp_5^ReadVariableOp_8*
bodyR
while_body_3809*9
_output_shapes'
%: : : : :��������� : : : : : *8
output_shapes'
%: : : : :��������� : : : : : *
T
2
*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_3808*
_num_original_outputs
K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
_output_shapes
: *
T0^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:��������� M
while/Identity_5Identitywhile:output:5*
_output_shapes
: *
T0M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
_output_shapes
: *
T0M
while/Identity_8Identitywhile:output:8*
_output_shapes
: *
T0M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
valueB"����    *
dtype0�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:��������� i
strided_slice_12/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:b
strided_slice_12/stack_1Const*
_output_shapes
:*
valueB: *
dtype0b
strided_slice_12/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
strided_slice_12StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*'
_output_shapes
:��������� *
Index0*
T0*
shrink_axis_maske
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� �
IdentityIdentitystrided_slice_12:output:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8^while*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*6
_input_shapes%
#:���������:::2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82
whilewhile2 
ReadVariableOpReadVariableOp: :& "
 
_user_specified_nameinputs: : 
�
�
$__inference_dense_layer_call_fn_4805

inputs(
$statefulpartitionedcall_dense_kernel&
"statefulpartitionedcall_dense_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs$statefulpartitionedcall_dense_kernel"statefulpartitionedcall_dense_bias**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:���������*
Tin
2*+
_gradient_op_typePartitionedCall-2999*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_2992*
Tout
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:��������� ::22
StatefulPartitionedCallStatefulPartitionedCall: : :& "
 
_user_specified_nameinputs
�	
�
)__inference_sequential_layer_call_fn_3676

inputs/
+statefulpartitionedcall_features_embeddings&
"statefulpartitionedcall_gru_kernel$
 statefulpartitionedcall_gru_bias0
,statefulpartitionedcall_gru_recurrent_kernel(
$statefulpartitionedcall_dense_kernel&
"statefulpartitionedcall_dense_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs+statefulpartitionedcall_features_embeddings"statefulpartitionedcall_gru_kernel statefulpartitionedcall_gru_bias,statefulpartitionedcall_gru_recurrent_kernel$statefulpartitionedcall_dense_kernel"statefulpartitionedcall_dense_bias*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_3072*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
	2*'
_output_shapes
:���������*+
_gradient_op_typePartitionedCall-3073�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*>
_input_shapes-
+:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : 
�

�
)__inference_sequential_layer_call_fn_3082
features_input/
+statefulpartitionedcall_features_embeddings&
"statefulpartitionedcall_gru_kernel$
 statefulpartitionedcall_gru_bias0
,statefulpartitionedcall_gru_recurrent_kernel(
$statefulpartitionedcall_dense_kernel&
"statefulpartitionedcall_dense_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallfeatures_input+statefulpartitionedcall_features_embeddings"statefulpartitionedcall_gru_kernel statefulpartitionedcall_gru_bias,statefulpartitionedcall_gru_recurrent_kernel$statefulpartitionedcall_dense_kernel"statefulpartitionedcall_dense_bias**
config_proto

GPU 

CPU2J 8*
Tin
	2*'
_output_shapes
:���������*+
_gradient_op_typePartitionedCall-3073*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_3072*
Tout
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*>
_input_shapes-
+:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall:. *
(
_user_specified_namefeatures_input: : : : : : 
�j
�
while_body_3809
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_gru_kernel_0
readvariableop_3_gru_bias_0+
'readvariableop_6_gru_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
readvariableop_gru_kernel
readvariableop_3_gru_bias)
%readvariableop_6_gru_recurrent_kernel��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�ReadVariableOp_7�ReadVariableOp_8�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:���������j
ReadVariableOpReadVariableOpreadvariableop_gru_kernel_0*
dtype0*
_output_shapes

:`d
strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

: �
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice:output:0*
T0*'
_output_shapes
:��������� }
ReadVariableOp_1ReadVariableOpreadvariableop_gru_kernel_0^ReadVariableOp*
_output_shapes

:`*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
valueB"        *
dtype0h
strided_slice_2/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_2StridedSliceReadVariableOp_1:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
Index0*
T0�
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_2:output:0*'
_output_shapes
:��������� *
T0
ReadVariableOp_2ReadVariableOpreadvariableop_gru_kernel_0^ReadVariableOp_1*
_output_shapes

:`*
dtype0f
strided_slice_3/stackConst*
dtype0*
_output_shapes
:*
valueB"    @   h
strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB"        h
strided_slice_3/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_3StridedSliceReadVariableOp_2:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
T0*
Index0�
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_3:output:0*'
_output_shapes
:��������� *
T0h
ReadVariableOp_3ReadVariableOpreadvariableop_3_gru_bias_0*
dtype0*
_output_shapes
:`_
strided_slice_4/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_4/stack_1Const*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_4/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
strided_slice_4StridedSliceReadVariableOp_3:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*

begin_mask*
_output_shapes
: p
BiasAddBiasAddMatMul:product:0strided_slice_4:output:0*'
_output_shapes
:��������� *
T0{
ReadVariableOp_4ReadVariableOpreadvariableop_3_gru_bias_0^ReadVariableOp_3*
dtype0*
_output_shapes
:`_
strided_slice_5/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_5/stack_1Const*
valueB:@*
dtype0*
_output_shapes
:a
strided_slice_5/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
strided_slice_5StridedSliceReadVariableOp_4:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: t
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:��������� {
ReadVariableOp_5ReadVariableOpreadvariableop_3_gru_bias_0^ReadVariableOp_4*
dtype0*
_output_shapes
:`_
strided_slice_6/stackConst*
valueB:@*
dtype0*
_output_shapes
:a
strided_slice_6/stack_1Const*
_output_shapes
:*
valueB: *
dtype0a
strided_slice_6/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
strided_slice_6StridedSliceReadVariableOp_5:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
end_mask*
_output_shapes
: *
Index0*
T0t
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_6:output:0*'
_output_shapes
:��������� *
T0x
ReadVariableOp_6ReadVariableOp'readvariableop_6_gru_recurrent_kernel_0*
dtype0*
_output_shapes

: `f
strided_slice_7/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_7/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_7/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_7StridedSliceReadVariableOp_6:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
T0*
Index0m
MatMul_3MatMulplaceholder_2strided_slice_7:output:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_7ReadVariableOp'readvariableop_6_gru_recurrent_kernel_0^ReadVariableOp_6*
dtype0*
_output_shapes

: `f
strided_slice_8/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_8/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_8/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
strided_slice_8StridedSliceReadVariableOp_7:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
end_mask*
_output_shapes

:  *
Index0*
T0*

begin_maskm
MatMul_4MatMulplaceholder_2strided_slice_8:output:0*'
_output_shapes
:��������� *
T0d
addAddV2BiasAdd:output:0MatMul_3:product:0*'
_output_shapes
:��������� *
T0J
ConstConst*
_output_shapes
: *
valueB
 *��L>*
dtype0L
Const_1Const*
valueB
 *   ?*
dtype0*
_output_shapes
: U
MulMuladd:z:0Const:output:0*'
_output_shapes
:��������� *
T0Y
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:��������� \
clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:��������� T
clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:��������� h
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:��������� L
Const_2Const*
_output_shapes
: *
valueB
 *��L>*
dtype0L
Const_3Const*
dtype0*
_output_shapes
: *
valueB
 *   ?[
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:��������� [
Add_3Add	Mul_1:z:0Const_3:output:0*'
_output_shapes
:��������� *
T0^
clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*'
_output_shapes
:��������� *
T0V
clip_by_value_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*'
_output_shapes
:��������� *
T0b
mul_2Mulclip_by_value_1:z:0placeholder_2*
T0*'
_output_shapes
:��������� �
ReadVariableOp_8ReadVariableOp'readvariableop_6_gru_recurrent_kernel_0^ReadVariableOp_7*
dtype0*
_output_shapes

: `f
strided_slice_9/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_9/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_9/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_9StridedSliceReadVariableOp_8:value:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:  i
MatMul_5MatMul	mul_2:z:0strided_slice_9:output:0*'
_output_shapes
:��������� *
T0h
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_4:z:0*'
_output_shapes
:��������� *
T0`
mul_3Mulclip_by_value:z:0placeholder_2*
T0*'
_output_shapes
:��������� J
sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: _
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:��������� Q
mul_4Mulsub:z:0Tanh:y:0*'
_output_shapes
:��������� *
T0V
add_5AddV2	mul_3:z:0	mul_4:z:0*'
_output_shapes
:��������� *
T0�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_5:z:0*
element_dtype0*
_output_shapes
: I
add_6/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_6AddV2placeholderadd_6/y:output:0*
_output_shapes
: *
T0I
add_7/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_7AddV2while_loop_counteradd_7/y:output:0*
_output_shapes
: *
T0�
IdentityIdentity	add_7:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: �

Identity_1Identitywhile_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: �

Identity_2Identity	add_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: �

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: �

Identity_4Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*'
_output_shapes
:��������� "$
strided_slice_1strided_slice_1_0"P
%readvariableop_6_gru_recurrent_kernel'readvariableop_6_gru_recurrent_kernel_0"8
readvariableop_gru_kernelreadvariableop_gru_kernel_0"8
readvariableop_3_gru_biasreadvariableop_3_gru_bias_0"!

identity_1Identity_1:output:0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*>
_input_shapes-
+: : : : :��������� : : :::2$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_4: : : : : :	 :  : : : 
�j
�
while_body_4075
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_gru_kernel_0
readvariableop_3_gru_bias_0+
'readvariableop_6_gru_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
readvariableop_gru_kernel
readvariableop_3_gru_bias)
%readvariableop_6_gru_recurrent_kernel��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�ReadVariableOp_7�ReadVariableOp_8�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:���������j
ReadVariableOpReadVariableOpreadvariableop_gru_kernel_0*
dtype0*
_output_shapes

:`d
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"        f
strided_slice/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
end_mask*
_output_shapes

: *
T0*
Index0*

begin_mask�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice:output:0*
T0*'
_output_shapes
:��������� }
ReadVariableOp_1ReadVariableOpreadvariableop_gru_kernel_0^ReadVariableOp*
dtype0*
_output_shapes

:`f
strided_slice_2/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_2/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_2StridedSliceReadVariableOp_1:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

: �
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_2:output:0*'
_output_shapes
:��������� *
T0
ReadVariableOp_2ReadVariableOpreadvariableop_gru_kernel_0^ReadVariableOp_1*
dtype0*
_output_shapes

:`f
strided_slice_3/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_3/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
strided_slice_3StridedSliceReadVariableOp_2:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
end_mask*
_output_shapes

: *
Index0*
T0*

begin_mask�
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_3:output:0*
T0*'
_output_shapes
:��������� h
ReadVariableOp_3ReadVariableOpreadvariableop_3_gru_bias_0*
_output_shapes
:`*
dtype0_
strided_slice_4/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_4/stack_1Const*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_4/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_4StridedSliceReadVariableOp_3:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
T0*
Index0*

begin_mask*
_output_shapes
: p
BiasAddBiasAddMatMul:product:0strided_slice_4:output:0*'
_output_shapes
:��������� *
T0{
ReadVariableOp_4ReadVariableOpreadvariableop_3_gru_bias_0^ReadVariableOp_3*
_output_shapes
:`*
dtype0_
strided_slice_5/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_5/stack_1Const*
valueB:@*
dtype0*
_output_shapes
:a
strided_slice_5/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_5StridedSliceReadVariableOp_4:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
_output_shapes
: *
T0*
Index0t
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:��������� {
ReadVariableOp_5ReadVariableOpreadvariableop_3_gru_bias_0^ReadVariableOp_4*
dtype0*
_output_shapes
:`_
strided_slice_6/stackConst*
valueB:@*
dtype0*
_output_shapes
:a
strided_slice_6/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_6/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_6StridedSliceReadVariableOp_5:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
end_mask*
_output_shapes
: *
Index0*
T0t
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_6:output:0*
T0*'
_output_shapes
:��������� x
ReadVariableOp_6ReadVariableOp'readvariableop_6_gru_recurrent_kernel_0*
_output_shapes

: `*
dtype0f
strided_slice_7/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_7/stack_1Const*
_output_shapes
:*
valueB"        *
dtype0h
strided_slice_7/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
strided_slice_7StridedSliceReadVariableOp_6:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:  m
MatMul_3MatMulplaceholder_2strided_slice_7:output:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_7ReadVariableOp'readvariableop_6_gru_recurrent_kernel_0^ReadVariableOp_6*
dtype0*
_output_shapes

: `f
strided_slice_8/stackConst*
_output_shapes
:*
valueB"        *
dtype0h
strided_slice_8/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_8/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_8StridedSliceReadVariableOp_7:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:  m
MatMul_4MatMulplaceholder_2strided_slice_8:output:0*'
_output_shapes
:��������� *
T0d
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:��������� J
ConstConst*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_1Const*
_output_shapes
: *
valueB
 *   ?*
dtype0U
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:��������� Y
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:��������� \
clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*'
_output_shapes
:��������� *
T0T
clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*'
_output_shapes
:��������� *
T0h
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:��������� L
Const_2Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_3Const*
dtype0*
_output_shapes
: *
valueB
 *   ?[
Mul_1Mul	add_2:z:0Const_2:output:0*'
_output_shapes
:��������� *
T0[
Add_3Add	Mul_1:z:0Const_3:output:0*'
_output_shapes
:��������� *
T0^
clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:��������� V
clip_by_value_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*'
_output_shapes
:��������� *
T0b
mul_2Mulclip_by_value_1:z:0placeholder_2*
T0*'
_output_shapes
:��������� �
ReadVariableOp_8ReadVariableOp'readvariableop_6_gru_recurrent_kernel_0^ReadVariableOp_7*
dtype0*
_output_shapes

: `f
strided_slice_9/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_9/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_9/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_slice_9StridedSliceReadVariableOp_8:value:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
Index0*
T0i
MatMul_5MatMul	mul_2:z:0strided_slice_9:output:0*
T0*'
_output_shapes
:��������� h
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*'
_output_shapes
:��������� *
T0I
TanhTanh	add_4:z:0*'
_output_shapes
:��������� *
T0`
mul_3Mulclip_by_value:z:0placeholder_2*
T0*'
_output_shapes
:��������� J
sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?_
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:��������� Q
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:��������� �
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_5:z:0*
element_dtype0*
_output_shapes
: I
add_6/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_6AddV2placeholderadd_6/y:output:0*
_output_shapes
: *
T0I
add_7/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_7AddV2while_loop_counteradd_7/y:output:0*
T0*
_output_shapes
: �
IdentityIdentity	add_7:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: �

Identity_1Identitywhile_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: �

Identity_2Identity	add_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: �

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: �

Identity_4Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*'
_output_shapes
:��������� "!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"$
strided_slice_1strided_slice_1_0"P
%readvariableop_6_gru_recurrent_kernel'readvariableop_6_gru_recurrent_kernel_0"8
readvariableop_3_gru_biasreadvariableop_3_gru_bias_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"8
readvariableop_gru_kernelreadvariableop_gru_kernel_0"!

identity_1Identity_1:output:0*>
_input_shapes-
+: : : : :��������� : : :::2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82 
ReadVariableOpReadVariableOp: : : : : : :	 :  : : 
�
�
while_body_2320
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0(
$statefulpartitionedcall_gru_kernel_0&
"statefulpartitionedcall_gru_bias_02
.statefulpartitionedcall_gru_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor&
"statefulpartitionedcall_gru_kernel$
 statefulpartitionedcall_gru_bias0
,statefulpartitionedcall_gru_recurrent_kernel��StatefulPartitionedCall�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:����������
StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2$statefulpartitionedcall_gru_kernel_0"statefulpartitionedcall_gru_bias_0.statefulpartitionedcall_gru_recurrent_kernel_0*+
_gradient_op_typePartitionedCall-1954*K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_1934*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin	
2*:
_output_shapes(
&:��������� :��������� �
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder StatefulPartitionedCall:output:0*
element_dtype0*
_output_shapes
: G
add/yConst*
value	B :*
dtype0*
_output_shapes
: J
addAddV2placeholderadd/y:output:0*
_output_shapes
: *
T0I
add_1/yConst*
dtype0*
_output_shapes
: *
value	B :U
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: Z
IdentityIdentity	add_1:z:0^StatefulPartitionedCall*
_output_shapes
: *
T0k

Identity_1Identitywhile_maximum_iterations^StatefulPartitionedCall*
_output_shapes
: *
T0Z

Identity_2Identityadd:z:0^StatefulPartitionedCall*
T0*
_output_shapes
: �

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^StatefulPartitionedCall*
T0*
_output_shapes
: �

Identity_4Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"J
"statefulpartitionedcall_gru_kernel$statefulpartitionedcall_gru_kernel_0"!

identity_3Identity_3:output:0"^
,statefulpartitionedcall_gru_recurrent_kernel.statefulpartitionedcall_gru_recurrent_kernel_0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"F
 statefulpartitionedcall_gru_bias"statefulpartitionedcall_gru_bias_0"$
strided_slice_1strided_slice_1_0*>
_input_shapes-
+: : : : :��������� : : :::22
StatefulPartitionedCallStatefulPartitionedCall: : :	 :  : : : : : : 
�	
�
'__inference_gru_cell_layer_call_fn_4994

inputs
states_0&
"statefulpartitionedcall_gru_kernel$
 statefulpartitionedcall_gru_bias0
,statefulpartitionedcall_gru_recurrent_kernel
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0"statefulpartitionedcall_gru_kernel statefulpartitionedcall_gru_bias,statefulpartitionedcall_gru_recurrent_kernel*+
_gradient_op_typePartitionedCall-1939*K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_1842*
Tout
2**
config_proto

GPU 

CPU2J 8*:
_output_shapes(
&:��������� :��������� *
Tin	
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� �

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0"
identityIdentity:output:0*E
_input_shapes4
2:���������:��������� :::22
StatefulPartitionedCallStatefulPartitionedCall: : :& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0: 
�
�
D__inference_sequential_layer_call_and_return_conditional_losses_3072

inputs8
4features_statefulpartitionedcall_features_embeddings*
&gru_statefulpartitionedcall_gru_kernel(
$gru_statefulpartitionedcall_gru_bias4
0gru_statefulpartitionedcall_gru_recurrent_kernel.
*dense_statefulpartitionedcall_dense_kernel,
(dense_statefulpartitionedcall_dense_bias
identity��dense/StatefulPartitionedCall� features/StatefulPartitionedCall�gru/StatefulPartitionedCall�
 features/StatefulPartitionedCallStatefulPartitionedCallinputs4features_statefulpartitionedcall_features_embeddings*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:���������*+
_gradient_op_typePartitionedCall-2417*K
fFRD
B__inference_features_layer_call_and_return_conditional_losses_2410�
gru/StatefulPartitionedCallStatefulPartitionedCall)features/StatefulPartitionedCall:output:0&gru_statefulpartitionedcall_gru_kernel$gru_statefulpartitionedcall_gru_bias0gru_statefulpartitionedcall_gru_recurrent_kernel**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:��������� *
Tin
2*+
_gradient_op_typePartitionedCall-2970*F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_2958*
Tout
2�
dense/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0*dense_statefulpartitionedcall_dense_kernel(dense_statefulpartitionedcall_dense_bias*+
_gradient_op_typePartitionedCall-2999*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_2992*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:���������*
Tin
2�
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall!^features/StatefulPartitionedCall^gru/StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*>
_input_shapes-
+:���������::::::2D
 features/StatefulPartitionedCall features/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall: : :& "
 
_user_specified_nameinputs: : : : 
�
�
D__inference_sequential_layer_call_and_return_conditional_losses_3028
features_input8
4features_statefulpartitionedcall_features_embeddings*
&gru_statefulpartitionedcall_gru_kernel(
$gru_statefulpartitionedcall_gru_bias4
0gru_statefulpartitionedcall_gru_recurrent_kernel.
*dense_statefulpartitionedcall_dense_kernel,
(dense_statefulpartitionedcall_dense_bias
identity��dense/StatefulPartitionedCall� features/StatefulPartitionedCall�gru/StatefulPartitionedCall�
 features/StatefulPartitionedCallStatefulPartitionedCallfeatures_input4features_statefulpartitionedcall_features_embeddings*K
fFRD
B__inference_features_layer_call_and_return_conditional_losses_2410*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:���������*+
_gradient_op_typePartitionedCall-2417�
gru/StatefulPartitionedCallStatefulPartitionedCall)features/StatefulPartitionedCall:output:0&gru_statefulpartitionedcall_gru_kernel$gru_statefulpartitionedcall_gru_bias0gru_statefulpartitionedcall_gru_recurrent_kernel*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:��������� *
Tin
2*+
_gradient_op_typePartitionedCall-2970*F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_2958�
dense/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0*dense_statefulpartitionedcall_dense_kernel(dense_statefulpartitionedcall_dense_bias*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:���������*+
_gradient_op_typePartitionedCall-2999*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_2992�
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall!^features/StatefulPartitionedCall^gru/StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*>
_input_shapes-
+:���������::::::2D
 features/StatefulPartitionedCall features/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:. *
(
_user_specified_namefeatures_input: : : : : : 
�
�
gru_while_cond_3220
gru_while_loop_counter 
gru_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_gru_strided_slice_1/
+gru_tensorarrayunstack_tensorlistfromtensor

gru_kernel
gru_bias
gru_recurrent_kernel
identity
T
LessLessplaceholderless_gru_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*>
_input_shapes-
+: : : : :��������� : : ::::  : : : : : : : : :	 
�
�
while_cond_2319
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor

gru_kernel
gru_bias
gru_recurrent_kernel
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*>
_input_shapes-
+: : : : :��������� : : :::: : : :	 :  : : : : : 
�
�
B__inference_features_layer_call_and_return_conditional_losses_3685

inputs(
$embedding_lookup_features_embeddings
identity��embedding_lookupU
CastCastinputs*

SrcT0*

DstT0*'
_output_shapes
:����������
embedding_lookupResourceGather$embedding_lookup_features_embeddingsCast:y:0*
Tindices0*
dtype0*+
_output_shapes
:���������*7
_class-
+)loc:@embedding_lookup/features/embeddings�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*7
_class-
+)loc:@embedding_lookup/features/embeddings*+
_output_shapes
:����������
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*+
_output_shapes
:���������*
T0�
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:���������"
identityIdentity:output:0**
_input_shapes
:���������:2$
embedding_lookupembedding_lookup:& "
 
_user_specified_nameinputs: 
�U
�
B__inference_gru_cell_layer_call_and_return_conditional_losses_1842

inputs

states
readvariableop_gru_kernel
readvariableop_3_gru_bias)
%readvariableop_6_gru_recurrent_kernel
identity

identity_1��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�ReadVariableOp_7�ReadVariableOp_8h
ReadVariableOpReadVariableOpreadvariableop_gru_kernel*
_output_shapes

:`*
dtype0d
strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

: b
MatMulMatMulinputsstrided_slice:output:0*
T0*'
_output_shapes
:��������� {
ReadVariableOp_1ReadVariableOpreadvariableop_gru_kernel^ReadVariableOp*
dtype0*
_output_shapes

:`f
strided_slice_1/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_1/stack_1Const*
_output_shapes
:*
valueB"    @   *
dtype0h
strided_slice_1/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
_output_shapes

: *
Index0*
T0*

begin_mask*
end_maskf
MatMul_1MatMulinputsstrided_slice_1:output:0*'
_output_shapes
:��������� *
T0}
ReadVariableOp_2ReadVariableOpreadvariableop_gru_kernel^ReadVariableOp_1*
dtype0*
_output_shapes

:`f
strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB"    @   h
strided_slice_2/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_2/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
end_mask*
_output_shapes

: *
Index0*
T0*

begin_maskf
MatMul_2MatMulinputsstrided_slice_2:output:0*
T0*'
_output_shapes
:��������� f
ReadVariableOp_3ReadVariableOpreadvariableop_3_gru_bias*
_output_shapes
:`*
dtype0_
strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
_output_shapes
:*
valueB: *
dtype0a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*

begin_mask*
_output_shapes
: p
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*'
_output_shapes
:��������� y
ReadVariableOp_4ReadVariableOpreadvariableop_3_gru_bias^ReadVariableOp_3*
_output_shapes
:`*
dtype0_
strided_slice_4/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_4/stack_1Const*
valueB:@*
dtype0*
_output_shapes
:a
strided_slice_4/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_4StridedSliceReadVariableOp_4:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
_output_shapes
: *
Index0*
T0t
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*'
_output_shapes
:��������� *
T0y
ReadVariableOp_5ReadVariableOpreadvariableop_3_gru_bias^ReadVariableOp_4*
dtype0*
_output_shapes
:`_
strided_slice_5/stackConst*
valueB:@*
dtype0*
_output_shapes
:a
strided_slice_5/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_5/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
strided_slice_5StridedSliceReadVariableOp_5:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
end_mask*
_output_shapes
: *
T0*
Index0t
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*'
_output_shapes
:��������� *
T0v
ReadVariableOp_6ReadVariableOp%readvariableop_6_gru_recurrent_kernel*
_output_shapes

: `*
dtype0f
strided_slice_6/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_6/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_6/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_slice_6StridedSliceReadVariableOp_6:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
T0*
Index0f
MatMul_3MatMulstatesstrided_slice_6:output:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_7ReadVariableOp%readvariableop_6_gru_recurrent_kernel^ReadVariableOp_6*
dtype0*
_output_shapes

: `f
strided_slice_7/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_7/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_7/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_slice_7StridedSliceReadVariableOp_7:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:  f
MatMul_4MatMulstatesstrided_slice_7:output:0*
T0*'
_output_shapes
:��������� d
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:��������� J
ConstConst*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *   ?U
MulMuladd:z:0Const:output:0*'
_output_shapes
:��������� *
T0Y
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:��������� \
clip_by_value/Minimum/yConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*'
_output_shapes
:��������� *
T0T
clip_by_value/yConst*
_output_shapes
: *
valueB
 *    *
dtype0
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:��������� h
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:��������� L
Const_2Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_3Const*
_output_shapes
: *
valueB
 *   ?*
dtype0[
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:��������� [
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:��������� ^
clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:��������� V
clip_by_value_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *    �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:��������� [
mul_2Mulclip_by_value_1:z:0states*
T0*'
_output_shapes
:��������� �
ReadVariableOp_8ReadVariableOp%readvariableop_6_gru_recurrent_kernel^ReadVariableOp_7*
dtype0*
_output_shapes

: `f
strided_slice_8/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_8/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_8/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_slice_8StridedSliceReadVariableOp_8:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
T0*
Index0i
MatMul_5MatMul	mul_2:z:0strided_slice_8:output:0*'
_output_shapes
:��������� *
T0h
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_4:z:0*'
_output_shapes
:��������� *
T0Y
mul_3Mulclip_by_value:z:0states*
T0*'
_output_shapes
:��������� J
sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: _
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:��������� Q
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_5AddV2	mul_3:z:0	mul_4:z:0*'
_output_shapes
:��������� *
T0�
IdentityIdentity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*'
_output_shapes
:��������� �

Identity_1Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0"
identityIdentity:output:0*E
_input_shapes4
2:���������:��������� :::2$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_4:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namestates: : : 
�
�
while_cond_4074
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor

gru_kernel
gru_bias
gru_recurrent_kernel
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*>
_input_shapes-
+: : : : :��������� : : ::::	 :  : : : : : : : : 
�k
�
sequential_gru_while_body_1563%
!sequential_gru_while_loop_counter+
'sequential_gru_while_maximum_iterations
placeholder
placeholder_1
placeholder_2$
 sequential_gru_strided_slice_1_0`
\tensorarrayv2read_tensorlistgetitem_sequential_gru_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_gru_kernel_0
readvariableop_3_gru_bias_0+
'readvariableop_6_gru_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4"
sequential_gru_strided_slice_1^
Ztensorarrayv2read_tensorlistgetitem_sequential_gru_tensorarrayunstack_tensorlistfromtensor
readvariableop_gru_kernel
readvariableop_3_gru_bias)
%readvariableop_6_gru_recurrent_kernel��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�ReadVariableOp_7�ReadVariableOp_8�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"����   �
#TensorArrayV2Read/TensorListGetItemTensorListGetItem\tensorarrayv2read_tensorlistgetitem_sequential_gru_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:���������j
ReadVariableOpReadVariableOpreadvariableop_gru_kernel_0*
dtype0*
_output_shapes

:`d
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"        f
strided_slice/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
Index0*
T0�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice:output:0*
T0*'
_output_shapes
:��������� }
ReadVariableOp_1ReadVariableOpreadvariableop_gru_kernel_0^ReadVariableOp*
dtype0*
_output_shapes

:`f
strided_slice_1/stackConst*
_output_shapes
:*
valueB"        *
dtype0h
strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB"    @   h
strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
end_mask*
_output_shapes

: *
Index0*
T0*

begin_mask�
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_1:output:0*
T0*'
_output_shapes
:��������� 
ReadVariableOp_2ReadVariableOpreadvariableop_gru_kernel_0^ReadVariableOp_1*
dtype0*
_output_shapes

:`f
strided_slice_2/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_2/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

: *
T0*
Index0�
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_2:output:0*
T0*'
_output_shapes
:��������� h
ReadVariableOp_3ReadVariableOpreadvariableop_3_gru_bias_0*
dtype0*
_output_shapes
:`_
strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*

begin_mask*
_output_shapes
: *
Index0*
T0p
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*'
_output_shapes
:��������� {
ReadVariableOp_4ReadVariableOpreadvariableop_3_gru_bias_0^ReadVariableOp_3*
dtype0*
_output_shapes
:`_
strided_slice_4/stackConst*
_output_shapes
:*
valueB: *
dtype0a
strided_slice_4/stack_1Const*
valueB:@*
dtype0*
_output_shapes
:a
strided_slice_4/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
strided_slice_4StridedSliceReadVariableOp_4:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
_output_shapes
: *
Index0*
T0t
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:��������� {
ReadVariableOp_5ReadVariableOpreadvariableop_3_gru_bias_0^ReadVariableOp_4*
dtype0*
_output_shapes
:`_
strided_slice_5/stackConst*
valueB:@*
dtype0*
_output_shapes
:a
strided_slice_5/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_5/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_5StridedSliceReadVariableOp_5:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
end_mask*
_output_shapes
: *
T0*
Index0t
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*'
_output_shapes
:��������� *
T0x
ReadVariableOp_6ReadVariableOp'readvariableop_6_gru_recurrent_kernel_0*
dtype0*
_output_shapes

: `f
strided_slice_6/stackConst*
_output_shapes
:*
valueB"        *
dtype0h
strided_slice_6/stack_1Const*
dtype0*
_output_shapes
:*
valueB"        h
strided_slice_6/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
strided_slice_6StridedSliceReadVariableOp_6:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:  m
MatMul_3MatMulplaceholder_2strided_slice_6:output:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_7ReadVariableOp'readvariableop_6_gru_recurrent_kernel_0^ReadVariableOp_6*
dtype0*
_output_shapes

: `f
strided_slice_7/stackConst*
_output_shapes
:*
valueB"        *
dtype0h
strided_slice_7/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_7/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_7StridedSliceReadVariableOp_7:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
end_mask*
_output_shapes

:  *
Index0*
T0*

begin_maskm
MatMul_4MatMulplaceholder_2strided_slice_7:output:0*'
_output_shapes
:��������� *
T0d
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:��������� J
ConstConst*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_1Const*
valueB
 *   ?*
dtype0*
_output_shapes
: U
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:��������� Y
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:��������� \
clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*'
_output_shapes
:��������� *
T0T
clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*'
_output_shapes
:��������� *
T0h
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:��������� L
Const_2Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_3Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_2:output:0*'
_output_shapes
:��������� *
T0[
Add_3Add	Mul_1:z:0Const_3:output:0*'
_output_shapes
:��������� *
T0^
clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:��������� V
clip_by_value_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*'
_output_shapes
:��������� *
T0b
mul_2Mulclip_by_value_1:z:0placeholder_2*'
_output_shapes
:��������� *
T0�
ReadVariableOp_8ReadVariableOp'readvariableop_6_gru_recurrent_kernel_0^ReadVariableOp_7*
dtype0*
_output_shapes

: `f
strided_slice_8/stackConst*
dtype0*
_output_shapes
:*
valueB"    @   h
strided_slice_8/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_8/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
strided_slice_8StridedSliceReadVariableOp_8:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
_output_shapes

:  *
Index0*
T0*

begin_mask*
end_maski
MatMul_5MatMul	mul_2:z:0strided_slice_8:output:0*
T0*'
_output_shapes
:��������� h
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*'
_output_shapes
:��������� *
T0I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:��������� `
mul_3Mulclip_by_value:z:0placeholder_2*'
_output_shapes
:��������� *
T0J
sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?_
subSubsub/x:output:0clip_by_value:z:0*'
_output_shapes
:��������� *
T0Q
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:��������� �
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_5:z:0*
element_dtype0*
_output_shapes
: I
add_6/yConst*
dtype0*
_output_shapes
: *
value	B :N
add_6AddV2placeholderadd_6/y:output:0*
T0*
_output_shapes
: I
add_7/yConst*
value	B :*
dtype0*
_output_shapes
: d
add_7AddV2!sequential_gru_while_loop_counteradd_7/y:output:0*
T0*
_output_shapes
: �
IdentityIdentity	add_7:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: �

Identity_1Identity'sequential_gru_while_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
_output_shapes
: *
T0�

Identity_2Identity	add_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: �

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: �

Identity_4Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*'
_output_shapes
:��������� *
T0"
identityIdentity:output:0"�
Ztensorarrayv2read_tensorlistgetitem_sequential_gru_tensorarrayunstack_tensorlistfromtensor\tensorarrayv2read_tensorlistgetitem_sequential_gru_tensorarrayunstack_tensorlistfromtensor_0"B
sequential_gru_strided_slice_1 sequential_gru_strided_slice_1_0"P
%readvariableop_6_gru_recurrent_kernel'readvariableop_6_gru_recurrent_kernel_0"8
readvariableop_3_gru_biasreadvariableop_3_gru_bias_0"8
readvariableop_gru_kernelreadvariableop_gru_kernel_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*>
_input_shapes-
+: : : : :��������� : : :::2$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_2:	 :  : : : : : : : : "�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*�
serving_default�
I
features_input7
 serving_default_features_input:0���������9
dense0
StatefulPartitionedCall:0���������tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:��
�$
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
trainable_variables
regularization_losses
	variables
		keras_api


signatures
a__call__
b_default_save_signature
*c&call_and_return_all_conditional_losses"�"
_tf_keras_sequential�!{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": null, "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "Embedding", "config": {"name": "features", "trainable": true, "batch_input_shape": [null, 30], "dtype": "float32", "input_dim": 124688, "output_dim": 25, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null, "dtype": "float32"}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 30}}, {"class_name": "GRU", "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 1, "reset_after": false}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 24, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": null, "activity_regularizer": null, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Embedding", "config": {"name": "features", "trainable": true, "batch_input_shape": [null, 30], "dtype": "float32", "input_dim": 124688, "output_dim": 25, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null, "dtype": "float32"}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 30}}, {"class_name": "GRU", "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 1, "reset_after": false}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 24, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�
trainable_variables
regularization_losses
	variables
	keras_api
d__call__
*e&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "InputLayer", "name": "features_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 30], "config": {"batch_input_shape": [null, 30], "dtype": "float32", "sparse": false, "ragged": false, "name": "features_input"}, "input_spec": null, "activity_regularizer": null}
�

embeddings
_callable_losses
trainable_variables
regularization_losses
	variables
	keras_api
f__call__
*g&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Embedding", "name": "features", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 30], "config": {"name": "features", "trainable": true, "batch_input_shape": [null, 30], "dtype": "float32", "input_dim": 124688, "output_dim": 25, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null, "dtype": "float32"}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 30}, "input_spec": null, "activity_regularizer": null}
�

cell

state_spec
_callable_losses
trainable_variables
regularization_losses
	variables
	keras_api
h__call__
*i&call_and_return_all_conditional_losses"�	
_tf_keras_layer�	{"class_name": "GRU", "name": "gru", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 1, "reset_after": false}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 25], "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "activity_regularizer": null}
�

kernel
bias
_callable_losses
trainable_variables
 regularization_losses
!	variables
"	keras_api
j__call__
*k&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 24, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "activity_regularizer": null}
�
#iter

$beta_1

%beta_2
	&decay
'learning_ratemUmVmW(mX)mY*mZv[v\v](v^)v_*v`"
	optimizer
J
0
(1
)2
*3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
(1
)2
*3
4
5"
trackable_list_wrapper
�
+layer_regularization_losses
trainable_variables
,metrics
regularization_losses
-non_trainable_variables

.layers
	variables
a__call__
b_default_save_signature
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
,
lserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
/layer_regularization_losses
trainable_variables
0metrics
regularization_losses
1non_trainable_variables

2layers
	variables
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
':%
��2features/embeddings
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
�
3layer_regularization_losses
trainable_variables
4metrics
regularization_losses
5non_trainable_variables

6layers
	variables
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
�

(kernel
)recurrent_kernel
*bias
7_callable_losses
8trainable_variables
9regularization_losses
:	variables
;	keras_api
m__call__
*n&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "GRUCell", "name": "gru_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "gru_cell", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 1, "reset_after": false}, "input_spec": null, "activity_regularizer": null}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
�
<layer_regularization_losses
trainable_variables
=metrics
regularization_losses
>non_trainable_variables

?layers
	variables
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
: 2dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
@layer_regularization_losses
trainable_variables
Ametrics
 regularization_losses
Bnon_trainable_variables

Clayers
!	variables
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
:	 (2training/Adam/iter
: (2training/Adam/beta_1
: (2training/Adam/beta_2
: (2training/Adam/decay
%:# (2training/Adam/learning_rate
:`2
gru/kernel
&:$ `2gru/recurrent_kernel
:`2gru/bias
 "
trackable_list_wrapper
'
D0"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
�
Elayer_regularization_losses
8trainable_variables
Fmetrics
9regularization_losses
Gnon_trainable_variables

Hlayers
:	variables
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	Itotal
	Jcount
K
_fn_kwargs
L_updates
Mtrainable_variables
Nregularization_losses
O	variables
P	keras_api
o__call__
*p&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MeanMetricWrapper", "name": "acc", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "acc", "dtype": "float32"}, "input_spec": null, "activity_regularizer": null}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count_3
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
�
Qlayer_regularization_losses
Mtrainable_variables
Rmetrics
Nregularization_losses
Snon_trainable_variables

Tlayers
O	variables
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
5:3
��2#training/Adam/features/embeddings/m
,:* 2training/Adam/dense/kernel/m
&:$2training/Adam/dense/bias/m
*:(`2training/Adam/gru/kernel/m
4:2 `2$training/Adam/gru/recurrent_kernel/m
$:"`2training/Adam/gru/bias/m
5:3
��2#training/Adam/features/embeddings/v
,:* 2training/Adam/dense/kernel/v
&:$2training/Adam/dense/bias/v
*:(`2training/Adam/gru/kernel/v
4:2 `2$training/Adam/gru/recurrent_kernel/v
$:"`2training/Adam/gru/bias/v
�2�
)__inference_sequential_layer_call_fn_3665
)__inference_sequential_layer_call_fn_3082
)__inference_sequential_layer_call_fn_3676
)__inference_sequential_layer_call_fn_3054�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
__inference__wrapped_model_1718�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *-�*
(�%
features_input���������
�2�
D__inference_sequential_layer_call_and_return_conditional_losses_3654
D__inference_sequential_layer_call_and_return_conditional_losses_3028
D__inference_sequential_layer_call_and_return_conditional_losses_3376
D__inference_sequential_layer_call_and_return_conditional_losses_3012�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2�
'__inference_features_layer_call_fn_3691�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_features_layer_call_and_return_conditional_losses_3685�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
"__inference_gru_layer_call_fn_4787
"__inference_gru_layer_call_fn_4231
"__inference_gru_layer_call_fn_4239
"__inference_gru_layer_call_fn_4779�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
=__inference_gru_layer_call_and_return_conditional_losses_4771
=__inference_gru_layer_call_and_return_conditional_losses_4505
=__inference_gru_layer_call_and_return_conditional_losses_3957
=__inference_gru_layer_call_and_return_conditional_losses_4223�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
$__inference_dense_layer_call_fn_4805�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
?__inference_dense_layer_call_and_return_conditional_losses_4798�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
8B6
"__inference_signature_wrapper_3095features_input
�2�
'__inference_gru_cell_layer_call_fn_5005
'__inference_gru_cell_layer_call_fn_4994�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
B__inference_gru_cell_layer_call_and_return_conditional_losses_4983
B__inference_gru_cell_layer_call_and_return_conditional_losses_4894�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 �
"__inference_signature_wrapper_3095�(*)I�F
� 
?�<
:
features_input(�%
features_input���������"-�*
(
dense�
dense����������
B__inference_features_layer_call_and_return_conditional_losses_3685_/�,
%�"
 �
inputs���������
� ")�&
�
0���������
� }
'__inference_features_layer_call_fn_3691R/�,
%�"
 �
inputs���������
� "�����������
"__inference_gru_layer_call_fn_4239`(*)?�<
5�2
$�!
inputs���������

 
p 

 
� "���������� �
?__inference_dense_layer_call_and_return_conditional_losses_4798\/�,
%�"
 �
inputs��������� 
� "%�"
�
0���������
� �
B__inference_gru_cell_layer_call_and_return_conditional_losses_4983�(*)\�Y
R�O
 �
inputs���������
'�$
"�
states/0��������� 
p 
� "R�O
H�E
�
0/0��������� 
$�!
�
0/1/0��������� 
� �
__inference__wrapped_model_1718p(*)7�4
-�*
(�%
features_input���������
� "-�*
(
dense�
dense����������
D__inference_sequential_layer_call_and_return_conditional_losses_3654h(*)7�4
-�*
 �
inputs���������
p 

 
� "%�"
�
0���������
� �
D__inference_sequential_layer_call_and_return_conditional_losses_3376h(*)7�4
-�*
 �
inputs���������
p

 
� "%�"
�
0���������
� �
'__inference_gru_cell_layer_call_fn_5005�(*)\�Y
R�O
 �
inputs���������
'�$
"�
states/0��������� 
p 
� "D�A
�
0��������� 
"�
�
1/0��������� �
)__inference_sequential_layer_call_fn_3054c(*)?�<
5�2
(�%
features_input���������
p

 
� "�����������
"__inference_gru_layer_call_fn_4779p(*)O�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "���������� �
"__inference_gru_layer_call_fn_4787p(*)O�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "���������� �
D__inference_sequential_layer_call_and_return_conditional_losses_3012p(*)?�<
5�2
(�%
features_input���������
p

 
� "%�"
�
0���������
� �
=__inference_gru_layer_call_and_return_conditional_losses_4223m(*)?�<
5�2
$�!
inputs���������

 
p 

 
� "%�"
�
0��������� 
� �
)__inference_sequential_layer_call_fn_3082c(*)?�<
5�2
(�%
features_input���������
p 

 
� "�����������
=__inference_gru_layer_call_and_return_conditional_losses_4505}(*)O�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "%�"
�
0��������� 
� �
=__inference_gru_layer_call_and_return_conditional_losses_3957m(*)?�<
5�2
$�!
inputs���������

 
p

 
� "%�"
�
0��������� 
� �
'__inference_gru_cell_layer_call_fn_4994�(*)\�Y
R�O
 �
inputs���������
'�$
"�
states/0��������� 
p
� "D�A
�
0��������� 
"�
�
1/0��������� �
D__inference_sequential_layer_call_and_return_conditional_losses_3028p(*)?�<
5�2
(�%
features_input���������
p 

 
� "%�"
�
0���������
� �
)__inference_sequential_layer_call_fn_3665[(*)7�4
-�*
 �
inputs���������
p

 
� "�����������
=__inference_gru_layer_call_and_return_conditional_losses_4771}(*)O�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "%�"
�
0��������� 
� �
B__inference_gru_cell_layer_call_and_return_conditional_losses_4894�(*)\�Y
R�O
 �
inputs���������
'�$
"�
states/0��������� 
p
� "R�O
H�E
�
0/0��������� 
$�!
�
0/1/0��������� 
� �
)__inference_sequential_layer_call_fn_3676[(*)7�4
-�*
 �
inputs���������
p 

 
� "�����������
"__inference_gru_layer_call_fn_4231`(*)?�<
5�2
$�!
inputs���������

 
p

 
� "���������� w
$__inference_dense_layer_call_fn_4805O/�,
%�"
 �
inputs��������� 
� "����������