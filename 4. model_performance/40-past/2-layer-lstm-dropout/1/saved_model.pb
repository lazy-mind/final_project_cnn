��3
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
shapeshape�"serve*1.15.02v1.15.0-rc3-22-g590d6eef7e8��1
�
features/embeddingsVarHandleOp*
dtype0*
_output_shapes
: *
shape:
��*$
shared_namefeatures/embeddings
}
'features/embeddings/Read/ReadVariableOpReadVariableOpfeatures/embeddings*
dtype0* 
_output_shapes
:
��
t
dense/kernelVarHandleOp*
shape
: *
shared_namedense/kernel*
dtype0*
_output_shapes
: 
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes

: 
l

dense/biasVarHandleOp*
shape:*
shared_name
dense/bias*
dtype0*
_output_shapes
: 
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:
x
dense_1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape
:*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:
p
dense_1/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
x
training/Adam/iterVarHandleOp*
shape: *#
shared_nametraining/Adam/iter*
dtype0	*
_output_shapes
: 
q
&training/Adam/iter/Read/ReadVariableOpReadVariableOptraining/Adam/iter*
dtype0	*
_output_shapes
: 
|
training/Adam/beta_1VarHandleOp*
dtype0*
_output_shapes
: *
shape: *%
shared_nametraining/Adam/beta_1
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
training/Adam/decayVarHandleOp*
dtype0*
_output_shapes
: *
shape: *$
shared_nametraining/Adam/decay
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
s
lstm/kernelVarHandleOp*
shared_namelstm/kernel*
dtype0*
_output_shapes
: *
shape:	�
l
lstm/kernel/Read/ReadVariableOpReadVariableOplstm/kernel*
dtype0*
_output_shapes
:	�
�
lstm/recurrent_kernelVarHandleOp*
_output_shapes
: *
shape:	@�*&
shared_namelstm/recurrent_kernel*
dtype0
�
)lstm/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm/recurrent_kernel*
dtype0*
_output_shapes
:	@�
k
	lstm/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*
shared_name	lstm/bias
d
lstm/bias/Read/ReadVariableOpReadVariableOp	lstm/bias*
dtype0*
_output_shapes	
:�
w
lstm_1/kernelVarHandleOp*
shared_namelstm_1/kernel*
dtype0*
_output_shapes
: *
shape:	@�
p
!lstm_1/kernel/Read/ReadVariableOpReadVariableOplstm_1/kernel*
_output_shapes
:	@�*
dtype0
�
lstm_1/recurrent_kernelVarHandleOp*(
shared_namelstm_1/recurrent_kernel*
dtype0*
_output_shapes
: *
shape:	 �
�
+lstm_1/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm_1/recurrent_kernel*
dtype0*
_output_shapes
:	 �
o
lstm_1/biasVarHandleOp*
_output_shapes
: *
shape:�*
shared_namelstm_1/bias*
dtype0
h
lstm_1/bias/Read/ReadVariableOpReadVariableOplstm_1/bias*
dtype0*
_output_shapes	
:�
^
totalVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
b
count_3VarHandleOp*
shared_name	count_3*
dtype0*
_output_shapes
: *
shape: 
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
dtype0*
_output_shapes
: 
�
#training/Adam/features/embeddings/mVarHandleOp*4
shared_name%#training/Adam/features/embeddings/m*
dtype0*
_output_shapes
: *
shape:
��
�
7training/Adam/features/embeddings/m/Read/ReadVariableOpReadVariableOp#training/Adam/features/embeddings/m*
dtype0* 
_output_shapes
:
��
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
training/Adam/dense_1/kernel/mVarHandleOp*
shape
:*/
shared_name training/Adam/dense_1/kernel/m*
dtype0*
_output_shapes
: 
�
2training/Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/kernel/m*
dtype0*
_output_shapes

:
�
training/Adam/dense_1/bias/mVarHandleOp*-
shared_nametraining/Adam/dense_1/bias/m*
dtype0*
_output_shapes
: *
shape:
�
0training/Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/bias/m*
dtype0*
_output_shapes
:
�
training/Adam/lstm/kernel/mVarHandleOp*,
shared_nametraining/Adam/lstm/kernel/m*
dtype0*
_output_shapes
: *
shape:	�
�
/training/Adam/lstm/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/lstm/kernel/m*
_output_shapes
:	�*
dtype0
�
%training/Adam/lstm/recurrent_kernel/mVarHandleOp*
shape:	@�*6
shared_name'%training/Adam/lstm/recurrent_kernel/m*
dtype0*
_output_shapes
: 
�
9training/Adam/lstm/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp%training/Adam/lstm/recurrent_kernel/m*
dtype0*
_output_shapes
:	@�
�
training/Adam/lstm/bias/mVarHandleOp**
shared_nametraining/Adam/lstm/bias/m*
dtype0*
_output_shapes
: *
shape:�
�
-training/Adam/lstm/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/lstm/bias/m*
dtype0*
_output_shapes	
:�
�
training/Adam/lstm_1/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:	@�*.
shared_nametraining/Adam/lstm_1/kernel/m
�
1training/Adam/lstm_1/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/lstm_1/kernel/m*
dtype0*
_output_shapes
:	@�
�
'training/Adam/lstm_1/recurrent_kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:	 �*8
shared_name)'training/Adam/lstm_1/recurrent_kernel/m
�
;training/Adam/lstm_1/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp'training/Adam/lstm_1/recurrent_kernel/m*
dtype0*
_output_shapes
:	 �
�
training/Adam/lstm_1/bias/mVarHandleOp*,
shared_nametraining/Adam/lstm_1/bias/m*
dtype0*
_output_shapes
: *
shape:�
�
/training/Adam/lstm_1/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/lstm_1/bias/m*
dtype0*
_output_shapes	
:�
�
#training/Adam/features/embeddings/vVarHandleOp*
shape:
��*4
shared_name%#training/Adam/features/embeddings/v*
dtype0*
_output_shapes
: 
�
7training/Adam/features/embeddings/v/Read/ReadVariableOpReadVariableOp#training/Adam/features/embeddings/v*
dtype0* 
_output_shapes
:
��
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
training/Adam/dense/bias/vVarHandleOp*+
shared_nametraining/Adam/dense/bias/v*
dtype0*
_output_shapes
: *
shape:
�
.training/Adam/dense/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense/bias/v*
dtype0*
_output_shapes
:
�
training/Adam/dense_1/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape
:*/
shared_name training/Adam/dense_1/kernel/v
�
2training/Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/kernel/v*
dtype0*
_output_shapes

:
�
training/Adam/dense_1/bias/vVarHandleOp*-
shared_nametraining/Adam/dense_1/bias/v*
dtype0*
_output_shapes
: *
shape:
�
0training/Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/bias/v*
dtype0*
_output_shapes
:
�
training/Adam/lstm/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:	�*,
shared_nametraining/Adam/lstm/kernel/v
�
/training/Adam/lstm/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/lstm/kernel/v*
dtype0*
_output_shapes
:	�
�
%training/Adam/lstm/recurrent_kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:	@�*6
shared_name'%training/Adam/lstm/recurrent_kernel/v
�
9training/Adam/lstm/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp%training/Adam/lstm/recurrent_kernel/v*
dtype0*
_output_shapes
:	@�
�
training/Adam/lstm/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:�**
shared_nametraining/Adam/lstm/bias/v
�
-training/Adam/lstm/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/lstm/bias/v*
dtype0*
_output_shapes	
:�
�
training/Adam/lstm_1/kernel/vVarHandleOp*
shape:	@�*.
shared_nametraining/Adam/lstm_1/kernel/v*
dtype0*
_output_shapes
: 
�
1training/Adam/lstm_1/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/lstm_1/kernel/v*
dtype0*
_output_shapes
:	@�
�
'training/Adam/lstm_1/recurrent_kernel/vVarHandleOp*
shape:	 �*8
shared_name)'training/Adam/lstm_1/recurrent_kernel/v*
dtype0*
_output_shapes
: 
�
;training/Adam/lstm_1/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp'training/Adam/lstm_1/recurrent_kernel/v*
dtype0*
_output_shapes
:	 �
�
training/Adam/lstm_1/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*,
shared_nametraining/Adam/lstm_1/bias/v
�
/training/Adam/lstm_1/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/lstm_1/bias/v*
dtype0*
_output_shapes	
:�

NoOpNoOp
�E
ConstConst"/device:CPU:0*�E
value�EB�D B�D
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
	optimizer
	regularization_losses

trainable_variables
	variables
	keras_api

signatures
R
regularization_losses
trainable_variables
	variables
	keras_api
x

embeddings
_callable_losses
regularization_losses
trainable_variables
	variables
	keras_api
�
cell

state_spec
_callable_losses
regularization_losses
trainable_variables
	variables
	keras_api
�
cell
 
state_spec
!_callable_losses
"regularization_losses
#trainable_variables
$	variables
%	keras_api
~

&kernel
'bias
(_callable_losses
)regularization_losses
*trainable_variables
+	variables
,	keras_api
h
-_callable_losses
.regularization_losses
/trainable_variables
0	variables
1	keras_api
~

2kernel
3bias
4_callable_losses
5regularization_losses
6trainable_variables
7	variables
8	keras_api
�
9iter

:beta_1

;beta_2
	<decay
=learning_ratem�&m�'m�2m�3m�>m�?m�@m�Am�Bm�Cm�v�&v�'v�2v�3v�>v�?v�@v�Av�Bv�Cv�
 
N
0
>1
?2
@3
A4
B5
C6
&7
'8
29
310
N
0
>1
?2
@3
A4
B5
C6
&7
'8
29
310
�
Dlayer_regularization_losses
	regularization_losses

Elayers
Fmetrics

trainable_variables
	variables
Gnon_trainable_variables
 
 
 
 
�
Hlayer_regularization_losses
regularization_losses

Ilayers
Jmetrics
trainable_variables
	variables
Knon_trainable_variables
ca
VARIABLE_VALUEfeatures/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 
 

0

0
�
Llayer_regularization_losses
regularization_losses

Mlayers
Nmetrics
trainable_variables
	variables
Onon_trainable_variables
�

>kernel
?recurrent_kernel
@bias
P_callable_losses
Qregularization_losses
Rtrainable_variables
S	variables
T	keras_api
 
 
 

>0
?1
@2

>0
?1
@2
�
Ulayer_regularization_losses
regularization_losses

Vlayers
Wmetrics
trainable_variables
	variables
Xnon_trainable_variables
�

Akernel
Brecurrent_kernel
Cbias
Y_callable_losses
Zregularization_losses
[trainable_variables
\	variables
]	keras_api
 
 
 

A0
B1
C2

A0
B1
C2
�
^layer_regularization_losses
"regularization_losses

_layers
`metrics
#trainable_variables
$	variables
anon_trainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

&0
'1

&0
'1
�
blayer_regularization_losses
)regularization_losses

clayers
dmetrics
*trainable_variables
+	variables
enon_trainable_variables
 
 
 
 
�
flayer_regularization_losses
.regularization_losses

glayers
hmetrics
/trainable_variables
0	variables
inon_trainable_variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

20
31

20
31
�
jlayer_regularization_losses
5regularization_losses

klayers
lmetrics
6trainable_variables
7	variables
mnon_trainable_variables
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
QO
VARIABLE_VALUElstm/kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUElstm/recurrent_kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUE	lstm/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUElstm_1/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUElstm_1/recurrent_kernel0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUElstm_1/bias0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
 
*
0
1
2
3
4
5

n0
 
 
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
>0
?1
@2

>0
?1
@2
�
olayer_regularization_losses
Qregularization_losses

players
qmetrics
Rtrainable_variables
S	variables
rnon_trainable_variables
 

0
 
 
 
 

A0
B1
C2

A0
B1
C2
�
slayer_regularization_losses
Zregularization_losses

tlayers
umetrics
[trainable_variables
\	variables
vnon_trainable_variables
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
�
	wtotal
	xcount
y
_fn_kwargs
z_updates
{regularization_losses
|trainable_variables
}	variables
~	keras_api
 
 
 
 
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
w0
x1
�
layer_regularization_losses
{regularization_losses
�layers
�metrics
|trainable_variables
}	variables
�non_trainable_variables
 
 
 

w0
x1
��
VARIABLE_VALUE#training/Adam/features/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining/Adam/dense/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEtraining/Adam/dense/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining/Adam/dense_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining/Adam/dense_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEtraining/Adam/lstm/kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE%training/Adam/lstm/recurrent_kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEtraining/Adam/lstm/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEtraining/Adam/lstm_1/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE'training/Adam/lstm_1/recurrent_kernel/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEtraining/Adam/lstm_1/bias/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#training/Adam/features/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining/Adam/dense/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEtraining/Adam/dense/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining/Adam/dense_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining/Adam/dense_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEtraining/Adam/lstm/kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE%training/Adam/lstm/recurrent_kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEtraining/Adam/lstm/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEtraining/Adam/lstm_1/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE'training/Adam/lstm_1/recurrent_kernel/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEtraining/Adam/lstm_1/bias/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 
�
serving_default_features_inputPlaceholder*
dtype0*'
_output_shapes
:���������(*
shape:���������(
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_features_inputfeatures/embeddingslstm/kernel	lstm/biaslstm/recurrent_kernellstm_1/kernellstm_1/biaslstm_1/recurrent_kerneldense/kernel
dense/biasdense_1/kerneldense_1/bias*,
_gradient_op_typePartitionedCall-10271*+
f&R$
"__inference_signature_wrapper_6187*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:���������*
Tin
2
O
saver_filenamePlaceholder*
shape: *
dtype0*
_output_shapes
: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'features/embeddings/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp&training/Adam/iter/Read/ReadVariableOp(training/Adam/beta_1/Read/ReadVariableOp(training/Adam/beta_2/Read/ReadVariableOp'training/Adam/decay/Read/ReadVariableOp/training/Adam/learning_rate/Read/ReadVariableOplstm/kernel/Read/ReadVariableOp)lstm/recurrent_kernel/Read/ReadVariableOplstm/bias/Read/ReadVariableOp!lstm_1/kernel/Read/ReadVariableOp+lstm_1/recurrent_kernel/Read/ReadVariableOplstm_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount_3/Read/ReadVariableOp7training/Adam/features/embeddings/m/Read/ReadVariableOp0training/Adam/dense/kernel/m/Read/ReadVariableOp.training/Adam/dense/bias/m/Read/ReadVariableOp2training/Adam/dense_1/kernel/m/Read/ReadVariableOp0training/Adam/dense_1/bias/m/Read/ReadVariableOp/training/Adam/lstm/kernel/m/Read/ReadVariableOp9training/Adam/lstm/recurrent_kernel/m/Read/ReadVariableOp-training/Adam/lstm/bias/m/Read/ReadVariableOp1training/Adam/lstm_1/kernel/m/Read/ReadVariableOp;training/Adam/lstm_1/recurrent_kernel/m/Read/ReadVariableOp/training/Adam/lstm_1/bias/m/Read/ReadVariableOp7training/Adam/features/embeddings/v/Read/ReadVariableOp0training/Adam/dense/kernel/v/Read/ReadVariableOp.training/Adam/dense/bias/v/Read/ReadVariableOp2training/Adam/dense_1/kernel/v/Read/ReadVariableOp0training/Adam/dense_1/bias/v/Read/ReadVariableOp/training/Adam/lstm/kernel/v/Read/ReadVariableOp9training/Adam/lstm/recurrent_kernel/v/Read/ReadVariableOp-training/Adam/lstm/bias/v/Read/ReadVariableOp1training/Adam/lstm_1/kernel/v/Read/ReadVariableOp;training/Adam/lstm_1/recurrent_kernel/v/Read/ReadVariableOp/training/Adam/lstm_1/bias/v/Read/ReadVariableOpConst**
config_proto

GPU 

CPU2J 8*
_output_shapes
: *5
Tin.
,2*	*,
_gradient_op_typePartitionedCall-10333*'
f"R 
__inference__traced_save_10332*
Tout
2
�

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamefeatures/embeddingsdense/kernel
dense/biasdense_1/kerneldense_1/biastraining/Adam/itertraining/Adam/beta_1training/Adam/beta_2training/Adam/decaytraining/Adam/learning_ratelstm/kernellstm/recurrent_kernel	lstm/biaslstm_1/kernellstm_1/recurrent_kernellstm_1/biastotalcount_3#training/Adam/features/embeddings/mtraining/Adam/dense/kernel/mtraining/Adam/dense/bias/mtraining/Adam/dense_1/kernel/mtraining/Adam/dense_1/bias/mtraining/Adam/lstm/kernel/m%training/Adam/lstm/recurrent_kernel/mtraining/Adam/lstm/bias/mtraining/Adam/lstm_1/kernel/m'training/Adam/lstm_1/recurrent_kernel/mtraining/Adam/lstm_1/bias/m#training/Adam/features/embeddings/vtraining/Adam/dense/kernel/vtraining/Adam/dense/bias/vtraining/Adam/dense_1/kernel/vtraining/Adam/dense_1/bias/vtraining/Adam/lstm/kernel/v%training/Adam/lstm/recurrent_kernel/vtraining/Adam/lstm/bias/vtraining/Adam/lstm_1/kernel/v'training/Adam/lstm_1/recurrent_kernel/vtraining/Adam/lstm_1/bias/v*,
_gradient_op_typePartitionedCall-10466**
f%R#
!__inference__traced_restore_10465*
Tout
2**
config_proto

GPU 

CPU2J 8*
_output_shapes
: *4
Tin-
+2)��/
�`
�
lstm_while_body_6321
lstm_while_loop_counter!
lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
lstm_strided_slice_1_0V
Rtensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0&
"split_readvariableop_lstm_kernel_0&
"split_1_readvariableop_lstm_bias_0*
&readvariableop_lstm_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
lstm_strided_slice_1T
Ptensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor$
 split_readvariableop_lstm_kernel$
 split_1_readvariableop_lstm_bias(
$readvariableop_lstm_recurrent_kernel��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemRtensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:���������G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: x
split/ReadVariableOpReadVariableOp"split_readvariableop_lstm_kernel_0*
dtype0*
_output_shapes
:	��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split*<
_output_shapes*
(:@:@:@:@~
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:0*
T0*'
_output_shapes
:���������@�
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:1*
T0*'
_output_shapes
:���������@�
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:2*
T0*'
_output_shapes
:���������@�
MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:3*
T0*'
_output_shapes
:���������@I
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: v
split_1/ReadVariableOpReadVariableOp"split_1_readvariableop_lstm_bias_0*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split*,
_output_shapes
:@:@:@:@h
BiasAddBiasAddMatMul:product:0split_1:output:0*'
_output_shapes
:���������@*
T0l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:���������@l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*'
_output_shapes
:���������@*
T0l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*'
_output_shapes
:���������@*
T0v
ReadVariableOpReadVariableOp&readvariableop_lstm_recurrent_kernel_0*
dtype0*
_output_shapes
:	@�d
strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_1Const*
valueB"    @   *
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

:@@*
T0*
Index0*

begin_maskk
MatMul_4MatMulplaceholder_2strided_slice:output:0*'
_output_shapes
:���������@*
T0d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:���������@L
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
: W
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:���������@Y
Add_1AddMul:z:0Const_3:output:0*'
_output_shapes
:���������@*
T0\
clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������@T
clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*'
_output_shapes
:���������@*
T0�
ReadVariableOp_1ReadVariableOp&readvariableop_lstm_recurrent_kernel_0^ReadVariableOp*
dtype0*
_output_shapes
:	@�f
strided_slice_1/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_1/stack_1Const*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:@@m
MatMul_5MatMulplaceholder_2strided_slice_1:output:0*
T0*'
_output_shapes
:���������@h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:���������@L
Const_4Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_4:output:0*'
_output_shapes
:���������@*
T0[
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*'
_output_shapes
:���������@^
clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*'
_output_shapes
:���������@*
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
:���������@b
mul_2Mulclip_by_value_1:z:0placeholder_3*'
_output_shapes
:���������@*
T0�
ReadVariableOp_2ReadVariableOp&readvariableop_lstm_recurrent_kernel_0^ReadVariableOp_1*
dtype0*
_output_shapes
:	@�f
strided_slice_2/stackConst*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_1Const*
valueB"    �   *
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
end_mask*
_output_shapes

:@@*
Index0*
T0m
MatMul_6MatMulplaceholder_2strided_slice_2:output:0*'
_output_shapes
:���������@*
T0h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*'
_output_shapes
:���������@*
T0I
TanhTanh	add_4:z:0*'
_output_shapes
:���������@*
T0[
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:���������@V
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:���������@�
ReadVariableOp_3ReadVariableOp&readvariableop_lstm_recurrent_kernel_0^ReadVariableOp_2*
dtype0*
_output_shapes
:	@�f
strided_slice_3/stackConst*
valueB"    �   *
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
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:@@m
MatMul_7MatMulplaceholder_2strided_slice_3:output:0*
T0*'
_output_shapes
:���������@h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*'
_output_shapes
:���������@*
T0L
Const_6Const*
dtype0*
_output_shapes
: *
valueB
 *��L>L
Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:���������@[
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:���������@^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
valueB
 *  �?*
dtype0�
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:���������@V
clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:���������@K
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:���������@_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*'
_output_shapes
:���������@*
T0�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_5:z:0*
element_dtype0*
_output_shapes
: I
add_8/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_8AddV2placeholderadd_8/y:output:0*
_output_shapes
: *
T0I
add_9/yConst*
value	B :*
dtype0*
_output_shapes
: Z
add_9AddV2lstm_while_loop_counteradd_9/y:output:0*
_output_shapes
: *
T0�
IdentityIdentity	add_9:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_1Identitylstm_while_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_2Identity	add_8:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_4Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*'
_output_shapes
:���������@*
T0�

Identity_5Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:���������@".
lstm_strided_slice_1lstm_strided_slice_1_0"F
 split_readvariableop_lstm_kernel"split_readvariableop_lstm_kernel_0"F
 split_1_readvariableop_lstm_bias"split_1_readvariableop_lstm_bias_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"�
Ptensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorRtensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0"N
$readvariableop_lstm_recurrent_kernel&readvariableop_lstm_recurrent_kernel_0"
identityIdentity:output:0"!

identity_5Identity_5:output:0*Q
_input_shapes@
>: : : : :���������@:���������@: : :::2 
ReadVariableOpReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp: : : : : : :	 :
 :  : : 
��
�
@__inference_lstm_1_layer_call_and_return_conditional_losses_9680

inputs&
"split_readvariableop_lstm_1_kernel&
"split_1_readvariableop_lstm_1_bias*
&readvariableop_lstm_1_recurrent_kernel
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOp�while;
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
zeros/mul/yConst*
dtype0*
_output_shapes
: *
value	B : _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
dtype0*
_output_shapes
: *
value
B :�Y

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
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� O
zeros_1/mul/yConst*
value	B : *
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B : *
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*!
valueB"          *
dtype0m
	transpose	Transposeinputstranspose/perm:output:0*+
_output_shapes
:(���������@*
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
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
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
valueB"����@   *
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
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*'
_output_shapes
:���������@*
Index0*
T0*
shrink_axis_maskG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: x
split/ReadVariableOpReadVariableOp"split_readvariableop_lstm_1_kernel*
dtype0*
_output_shapes
:	@��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
	num_split*<
_output_shapes*
(:@ :@ :@ :@ *
T0l
MatMulMatMulstrided_slice_2:output:0split:output:0*'
_output_shapes
:��������� *
T0n
MatMul_1MatMulstrided_slice_2:output:0split:output:1*
T0*'
_output_shapes
:��������� n
MatMul_2MatMulstrided_slice_2:output:0split:output:2*'
_output_shapes
:��������� *
T0n
MatMul_3MatMulstrided_slice_2:output:0split:output:3*
T0*'
_output_shapes
:��������� I
Const_1Const*
_output_shapes
: *
value	B :*
dtype0S
split_1/split_dimConst*
_output_shapes
: *
value	B : *
dtype0v
split_1/ReadVariableOpReadVariableOp"split_1_readvariableop_lstm_1_bias*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
	num_split*,
_output_shapes
: : : : *
T0h
BiasAddBiasAddMatMul:product:0split_1:output:0*'
_output_shapes
:��������� *
T0l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*'
_output_shapes
:��������� *
T0l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:��������� l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:��������� v
ReadVariableOpReadVariableOp&readvariableop_lstm_1_recurrent_kernel*
dtype0*
_output_shapes
:	 �f
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

:  *
Index0*
T0n
MatMul_4MatMulzeros:output:0strided_slice_3:output:0*
T0*'
_output_shapes
:��������� d
addAddV2BiasAdd:output:0MatMul_4:product:0*
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
: W
MulMuladd:z:0Const_2:output:0*'
_output_shapes
:��������� *
T0Y
Add_1AddMul:z:0Const_3:output:0*'
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
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*'
_output_shapes
:��������� *
T0�
ReadVariableOp_1ReadVariableOp&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp*
dtype0*
_output_shapes
:	 �f
strided_slice_4/stackConst*
_output_shapes
:*
valueB"        *
dtype0h
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

:  *
Index0*
T0n
MatMul_5MatMulzeros:output:0strided_slice_4:output:0*'
_output_shapes
:��������� *
T0h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*'
_output_shapes
:��������� *
T0L
Const_4Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:��������� [
Add_3Add	Mul_1:z:0Const_5:output:0*
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
:��������� e
mul_2Mulclip_by_value_1:z:0zeros_1:output:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_2ReadVariableOp&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp_1*
dtype0*
_output_shapes
:	 �f
strided_slice_5/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_5/stack_1Const*
valueB"    `   *
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
end_mask*
_output_shapes

:  n
MatMul_6MatMulzeros:output:0strided_slice_5:output:0*
T0*'
_output_shapes
:��������� h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:��������� [
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_5AddV2	mul_2:z:0	mul_3:z:0*'
_output_shapes
:��������� *
T0�
ReadVariableOp_3ReadVariableOp&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp_2*
_output_shapes
:	 �*
dtype0f
strided_slice_6/stackConst*
valueB"    `   *
dtype0*
_output_shapes
:h
strided_slice_6/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_6/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_6StridedSliceReadVariableOp_3:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:  n
MatMul_7MatMulzeros:output:0strided_slice_6:output:0*'
_output_shapes
:��������� *
T0h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:��������� L
Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:��������� [
Add_7Add	Mul_4:z:0Const_7:output:0*'
_output_shapes
:��������� *
T0^
clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:��������� V
clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:��������� K
Tanh_1Tanh	add_5:z:0*'
_output_shapes
:��������� *
T0_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*'
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
timeConst*
_output_shapes
: *
value	B : *
dtype0c
while/maximum_iterationsConst*
valueB :
���������*
dtype0*
_output_shapes
: T
while/loop_counterConst*
dtype0*
_output_shapes
: *
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"split_readvariableop_lstm_1_kernel"split_1_readvariableop_lstm_1_bias&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_9526*
_num_original_outputs*
bodyR
while_body_9527*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *K
output_shapes:
8: : : : :��������� :��������� : : : : : *
T
2K
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
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:��������� ^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:��������� M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
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
:(��������� h
strided_slice_7/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:a
strided_slice_7/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_7/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
strided_slice_7StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:��������� *
Index0*
T0e
transpose_1/permConst*
dtype0*
_output_shapes
:*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*+
_output_shapes
:���������( *
T0�
IdentityIdentitystrided_slice_7:output:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp^while*'
_output_shapes
:��������� *
T0"
identityIdentity:output:0*6
_input_shapes%
#:���������(@:::20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp2
whilewhile2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs: : : 
�
�
B__inference_features_layer_call_and_return_conditional_losses_7394

inputs(
$embedding_lookup_features_embeddings
identity��embedding_lookupU
CastCastinputs*

SrcT0*

DstT0*'
_output_shapes
:���������(�
embedding_lookupResourceGather$embedding_lookup_features_embeddingsCast:y:0*
Tindices0*
dtype0*+
_output_shapes
:���������(*7
_class-
+)loc:@embedding_lookup/features/embeddings�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*7
_class-
+)loc:@embedding_lookup/features/embeddings*+
_output_shapes
:���������(�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������(�
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*+
_output_shapes
:���������(*
T0"
identityIdentity:output:0**
_input_shapes
:���������(:2$
embedding_lookupembedding_lookup:& "
 
_user_specified_nameinputs: 
�
�
sequential_lstm_while_cond_2829&
"sequential_lstm_while_loop_counter,
(sequential_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3(
$less_sequential_lstm_strided_slice_1;
7sequential_lstm_tensorarrayunstack_tensorlistfromtensor
lstm_kernel
	lstm_bias
lstm_recurrent_kernel
identity
`
LessLessplaceholder$less_sequential_lstm_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :���������@:���������@: : :::: : : :	 :
 :  : : : : : 
�`
�
while_body_8953
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0(
$split_readvariableop_lstm_1_kernel_0(
$split_1_readvariableop_lstm_1_bias_0,
(readvariableop_lstm_1_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor&
"split_readvariableop_lstm_1_kernel&
"split_1_readvariableop_lstm_1_bias*
&readvariableop_lstm_1_recurrent_kernel��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����@   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:���������@G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: z
split/ReadVariableOpReadVariableOp$split_readvariableop_lstm_1_kernel_0*
dtype0*
_output_shapes
:	@��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
	num_split*<
_output_shapes*
(:@ :@ :@ :@ *
T0~
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:0*
T0*'
_output_shapes
:��������� �
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:1*'
_output_shapes
:��������� *
T0�
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:2*
T0*'
_output_shapes
:��������� �
MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:3*'
_output_shapes
:��������� *
T0I
Const_1Const*
_output_shapes
: *
value	B :*
dtype0S
split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: x
split_1/ReadVariableOpReadVariableOp$split_1_readvariableop_lstm_1_bias_0*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split*,
_output_shapes
: : : : h
BiasAddBiasAddMatMul:product:0split_1:output:0*'
_output_shapes
:��������� *
T0l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:��������� l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*'
_output_shapes
:��������� *
T0l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:��������� x
ReadVariableOpReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0*
dtype0*
_output_shapes
:	 �d
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
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
Index0*
T0k
MatMul_4MatMulplaceholder_2strided_slice:output:0*
T0*'
_output_shapes
:��������� d
addAddV2BiasAdd:output:0MatMul_4:product:0*'
_output_shapes
:��������� *
T0L
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
: W
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:��������� Y
Add_1AddMul:z:0Const_3:output:0*'
_output_shapes
:��������� *
T0\
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
:��������� �
ReadVariableOp_1ReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0^ReadVariableOp*
dtype0*
_output_shapes
:	 �f
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

:  m
MatMul_5MatMulplaceholder_2strided_slice_2:output:0*
T0*'
_output_shapes
:��������� h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*'
_output_shapes
:��������� *
T0L
Const_4Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_4:output:0*'
_output_shapes
:��������� *
T0[
Add_3Add	Mul_1:z:0Const_5:output:0*
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
:��������� b
mul_2Mulclip_by_value_1:z:0placeholder_3*
T0*'
_output_shapes
:��������� �
ReadVariableOp_2ReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0^ReadVariableOp_1*
dtype0*
_output_shapes
:	 �f
strided_slice_3/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_3/stack_1Const*
valueB"    `   *
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
end_mask*
_output_shapes

:  m
MatMul_6MatMulplaceholder_2strided_slice_3:output:0*
T0*'
_output_shapes
:��������� h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:��������� [
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_5AddV2	mul_2:z:0	mul_3:z:0*'
_output_shapes
:��������� *
T0�
ReadVariableOp_3ReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0^ReadVariableOp_2*
dtype0*
_output_shapes
:	 �f
strided_slice_4/stackConst*
valueB"    `   *
dtype0*
_output_shapes
:h
strided_slice_4/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_4StridedSliceReadVariableOp_3:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:  m
MatMul_7MatMulplaceholder_2strided_slice_4:output:0*
T0*'
_output_shapes
:��������� h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:��������� L
Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_4Mul	add_6:z:0Const_6:output:0*'
_output_shapes
:��������� *
T0[
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:��������� ^
clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*'
_output_shapes
:��������� *
T0V
clip_by_value_2/yConst*
dtype0*
_output_shapes
: *
valueB
 *    �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:��������� K
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:��������� _
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:��������� �
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_5:z:0*
element_dtype0*
_output_shapes
: I
add_8/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_8AddV2placeholderadd_8/y:output:0*
T0*
_output_shapes
: I
add_9/yConst*
dtype0*
_output_shapes
: *
value	B :U
add_9AddV2while_loop_counteradd_9/y:output:0*
T0*
_output_shapes
: �
IdentityIdentity	add_9:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_1Identitywhile_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_2Identity	add_8:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_4Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:��������� �

Identity_5Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:��������� "$
strided_slice_1strided_slice_1_0"J
"split_readvariableop_lstm_1_kernel$split_readvariableop_lstm_1_kernel_0"R
&readvariableop_lstm_1_recurrent_kernel(readvariableop_lstm_1_recurrent_kernel_0"!

identity_1Identity_1:output:0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"J
"split_1_readvariableop_lstm_1_bias$split_1_readvariableop_lstm_1_bias_0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0*Q
_input_shapes@
>: : : : :��������� :��������� : : :::2 
ReadVariableOpReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp:  : : : : : : : : :	 :
 
�L
�
C__inference_lstm_cell_layer_call_and_return_conditional_losses_3407

inputs

states
states_1$
 split_readvariableop_lstm_kernel$
 split_1_readvariableop_lstm_bias(
$readvariableop_lstm_recurrent_kernel
identity

identity_1

identity_2��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOpG
ConstConst*
dtype0*
_output_shapes
: *
value	B :Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: v
split/ReadVariableOpReadVariableOp split_readvariableop_lstm_kernel*
dtype0*
_output_shapes
:	��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split*<
_output_shapes*
(:@:@:@:@Z
MatMulMatMulinputssplit:output:0*'
_output_shapes
:���������@*
T0\
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:���������@\
MatMul_2MatMulinputssplit:output:2*'
_output_shapes
:���������@*
T0\
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:���������@I
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
dtype0*
_output_shapes
: *
value	B : t
split_1/ReadVariableOpReadVariableOp split_1_readvariableop_lstm_bias*
_output_shapes	
:�*
dtype0�
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split*,
_output_shapes
:@:@:@:@h
BiasAddBiasAddMatMul:product:0split_1:output:0*'
_output_shapes
:���������@*
T0l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*'
_output_shapes
:���������@*
T0l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:���������@l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:���������@t
ReadVariableOpReadVariableOp$readvariableop_lstm_recurrent_kernel*
dtype0*
_output_shapes
:	@�d
strided_slice/stackConst*
_output_shapes
:*
valueB"        *
dtype0f
strided_slice/stack_1Const*
_output_shapes
:*
valueB"    @   *
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

:@@*
Index0*
T0d
MatMul_4MatMulstatesstrided_slice:output:0*'
_output_shapes
:���������@*
T0d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:���������@L
Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *��L>L
Const_3Const*
valueB
 *   ?*
dtype0*
_output_shapes
: W
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:���������@Y
Add_1AddMul:z:0Const_3:output:0*
T0*'
_output_shapes
:���������@\
clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������@T
clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������@�
ReadVariableOp_1ReadVariableOp$readvariableop_lstm_recurrent_kernel^ReadVariableOp*
dtype0*
_output_shapes
:	@�f
strided_slice_1/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_1/stack_1Const*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_1/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
end_mask*
_output_shapes

:@@*
T0*
Index0*

begin_maskf
MatMul_5MatMulstatesstrided_slice_1:output:0*'
_output_shapes
:���������@*
T0h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*'
_output_shapes
:���������@*
T0L
Const_4Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:���������@[
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*'
_output_shapes
:���������@^
clip_by_value_1/Minimum/yConst*
dtype0*
_output_shapes
: *
valueB
 *  �?�
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:���������@V
clip_by_value_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:���������@]
mul_2Mulclip_by_value_1:z:0states_1*
T0*'
_output_shapes
:���������@�
ReadVariableOp_2ReadVariableOp$readvariableop_lstm_recurrent_kernel^ReadVariableOp_1*
dtype0*
_output_shapes
:	@�f
strided_slice_2/stackConst*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_1Const*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:@@f
MatMul_6MatMulstatesstrided_slice_2:output:0*
T0*'
_output_shapes
:���������@h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:���������@I
TanhTanh	add_4:z:0*'
_output_shapes
:���������@*
T0[
mul_3Mulclip_by_value:z:0Tanh:y:0*'
_output_shapes
:���������@*
T0V
add_5AddV2	mul_2:z:0	mul_3:z:0*'
_output_shapes
:���������@*
T0�
ReadVariableOp_3ReadVariableOp$readvariableop_lstm_recurrent_kernel^ReadVariableOp_2*
_output_shapes
:	@�*
dtype0f
strided_slice_3/stackConst*
valueB"    �   *
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
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:@@*
Index0*
T0f
MatMul_7MatMulstatesstrided_slice_3:output:0*
T0*'
_output_shapes
:���������@h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:���������@L
Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_7Const*
dtype0*
_output_shapes
: *
valueB
 *   ?[
Mul_4Mul	add_6:z:0Const_6:output:0*'
_output_shapes
:���������@*
T0[
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:���������@^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
valueB
 *  �?*
dtype0�
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:���������@V
clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*'
_output_shapes
:���������@*
T0K
Tanh_1Tanh	add_5:z:0*'
_output_shapes
:���������@*
T0_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*'
_output_shapes
:���������@*
T0�
IdentityIdentity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:���������@�

Identity_1Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:���������@�

Identity_2Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*X
_input_shapesG
E:���������:���������@:���������@:::2 
ReadVariableOpReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namestates:&"
 
_user_specified_namestates: : : 
�
�
while_cond_7525
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
lstm_kernel
	lstm_bias
lstm_recurrent_kernel
identity
P
LessLessplaceholderless_strided_slice_1*
_output_shapes
: *
T0?
IdentityIdentityLess:z:0*
_output_shapes
: *
T0
"
identityIdentity:output:0*Q
_input_shapes@
>: : : : :���������@:���������@: : ::::  : : : : : : : : :	 :
 
�!
�
D__inference_sequential_layer_call_and_return_conditional_losses_6065
features_input8
4features_statefulpartitionedcall_features_embeddings,
(lstm_statefulpartitionedcall_lstm_kernel*
&lstm_statefulpartitionedcall_lstm_bias6
2lstm_statefulpartitionedcall_lstm_recurrent_kernel0
,lstm_1_statefulpartitionedcall_lstm_1_kernel.
*lstm_1_statefulpartitionedcall_lstm_1_bias:
6lstm_1_statefulpartitionedcall_lstm_1_recurrent_kernel.
*dense_statefulpartitionedcall_dense_kernel,
(dense_statefulpartitionedcall_dense_bias2
.dense_1_statefulpartitionedcall_dense_1_kernel0
,dense_1_statefulpartitionedcall_dense_1_bias
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dropout/StatefulPartitionedCall� features/StatefulPartitionedCall�lstm/StatefulPartitionedCall�lstm_1/StatefulPartitionedCall�
 features/StatefulPartitionedCallStatefulPartitionedCallfeatures_input4features_statefulpartitionedcall_features_embeddings*
Tout
2**
config_proto

GPU 

CPU2J 8*+
_output_shapes
:���������(*
Tin
2*+
_gradient_op_typePartitionedCall-4788*K
fFRD
B__inference_features_layer_call_and_return_conditional_losses_4781�
lstm/StatefulPartitionedCallStatefulPartitionedCall)features/StatefulPartitionedCall:output:0(lstm_statefulpartitionedcall_lstm_kernel&lstm_statefulpartitionedcall_lstm_bias2lstm_statefulpartitionedcall_lstm_recurrent_kernel*+
_gradient_op_typePartitionedCall-5358*G
fBR@
>__inference_lstm_layer_call_and_return_conditional_losses_5076*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:���������(@�
lstm_1/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0,lstm_1_statefulpartitionedcall_lstm_1_kernel*lstm_1_statefulpartitionedcall_lstm_1_bias6lstm_1_statefulpartitionedcall_lstm_1_recurrent_kernel**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:��������� *+
_gradient_op_typePartitionedCall-5938*I
fDRB
@__inference_lstm_1_layer_call_and_return_conditional_losses_5656*
Tout
2�
dense/StatefulPartitionedCallStatefulPartitionedCall'lstm_1/StatefulPartitionedCall:output:0*dense_statefulpartitionedcall_dense_kernel(dense_statefulpartitionedcall_dense_bias*
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
_gradient_op_typePartitionedCall-5976*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_5969�
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_6008*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:���������*+
_gradient_op_typePartitionedCall-6020�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0.dense_1_statefulpartitionedcall_dense_1_kernel,dense_1_statefulpartitionedcall_dense_1_bias*
Tin
2*'
_output_shapes
:���������*+
_gradient_op_typePartitionedCall-6052*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_6045*
Tout
2**
config_proto

GPU 

CPU2J 8�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall!^features/StatefulPartitionedCall^lstm/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*'
_output_shapes
:���������*
T0"
identityIdentity:output:0*R
_input_shapesA
?:���������(:::::::::::2D
 features/StatefulPartitionedCall features/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:. *
(
_user_specified_namefeatures_input: : : : : : : : :	 :
 : 
�`
�
while_body_8100
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0&
"split_readvariableop_lstm_kernel_0&
"split_1_readvariableop_lstm_bias_0*
&readvariableop_lstm_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor$
 split_readvariableop_lstm_kernel$
 split_1_readvariableop_lstm_bias(
$readvariableop_lstm_recurrent_kernel��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:���������G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: x
split/ReadVariableOpReadVariableOp"split_readvariableop_lstm_kernel_0*
dtype0*
_output_shapes
:	��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split*<
_output_shapes*
(:@:@:@:@~
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:0*
T0*'
_output_shapes
:���������@�
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:1*
T0*'
_output_shapes
:���������@�
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:2*
T0*'
_output_shapes
:���������@�
MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:3*
T0*'
_output_shapes
:���������@I
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: v
split_1/ReadVariableOpReadVariableOp"split_1_readvariableop_lstm_bias_0*
_output_shapes	
:�*
dtype0�
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split*,
_output_shapes
:@:@:@:@h
BiasAddBiasAddMatMul:product:0split_1:output:0*'
_output_shapes
:���������@*
T0l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*'
_output_shapes
:���������@*
T0l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*'
_output_shapes
:���������@*
T0l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:���������@v
ReadVariableOpReadVariableOp&readvariableop_lstm_recurrent_kernel_0*
dtype0*
_output_shapes
:	@�d
strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_1Const*
valueB"    @   *
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

:@@*
T0*
Index0k
MatMul_4MatMulplaceholder_2strided_slice:output:0*'
_output_shapes
:���������@*
T0d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:���������@L
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
 *   ?W
MulMuladd:z:0Const_2:output:0*'
_output_shapes
:���������@*
T0Y
Add_1AddMul:z:0Const_3:output:0*'
_output_shapes
:���������@*
T0\
clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*'
_output_shapes
:���������@*
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
:���������@�
ReadVariableOp_1ReadVariableOp&readvariableop_lstm_recurrent_kernel_0^ReadVariableOp*
_output_shapes
:	@�*
dtype0f
strided_slice_2/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_1Const*
valueB"    �   *
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

:@@*
Index0*
T0*

begin_maskm
MatMul_5MatMulplaceholder_2strided_slice_2:output:0*'
_output_shapes
:���������@*
T0h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*'
_output_shapes
:���������@*
T0L
Const_4Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:���������@[
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*'
_output_shapes
:���������@^
clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*'
_output_shapes
:���������@*
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
:���������@b
mul_2Mulclip_by_value_1:z:0placeholder_3*
T0*'
_output_shapes
:���������@�
ReadVariableOp_2ReadVariableOp&readvariableop_lstm_recurrent_kernel_0^ReadVariableOp_1*
dtype0*
_output_shapes
:	@�f
strided_slice_3/stackConst*
_output_shapes
:*
valueB"    �   *
dtype0h
strided_slice_3/stack_1Const*
valueB"    �   *
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
end_mask*
_output_shapes

:@@m
MatMul_6MatMulplaceholder_2strided_slice_3:output:0*
T0*'
_output_shapes
:���������@h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:���������@I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:���������@[
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:���������@V
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:���������@�
ReadVariableOp_3ReadVariableOp&readvariableop_lstm_recurrent_kernel_0^ReadVariableOp_2*
dtype0*
_output_shapes
:	@�f
strided_slice_4/stackConst*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_4/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_4StridedSliceReadVariableOp_3:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:@@m
MatMul_7MatMulplaceholder_2strided_slice_4:output:0*'
_output_shapes
:���������@*
T0h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:���������@L
Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_4Mul	add_6:z:0Const_6:output:0*'
_output_shapes
:���������@*
T0[
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:���������@^
clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:���������@V
clip_by_value_2/yConst*
_output_shapes
: *
valueB
 *    *
dtype0�
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*'
_output_shapes
:���������@*
T0K
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:���������@_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:���������@�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_5:z:0*
element_dtype0*
_output_shapes
: I
add_8/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_8AddV2placeholderadd_8/y:output:0*
_output_shapes
: *
T0I
add_9/yConst*
dtype0*
_output_shapes
: *
value	B :U
add_9AddV2while_loop_counteradd_9/y:output:0*
T0*
_output_shapes
: �
IdentityIdentity	add_9:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_1Identitywhile_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_2Identity	add_8:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_4Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*'
_output_shapes
:���������@*
T0�

Identity_5Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:���������@"F
 split_readvariableop_lstm_kernel"split_readvariableop_lstm_kernel_0"$
strided_slice_1strided_slice_1_0"F
 split_1_readvariableop_lstm_bias"split_1_readvariableop_lstm_bias_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"N
$readvariableop_lstm_recurrent_kernel&readvariableop_lstm_recurrent_kernel_0"
identityIdentity:output:0"!

identity_5Identity_5:output:0*Q
_input_shapes@
>: : : : :���������@:���������@: : :::20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp2 
ReadVariableOpReadVariableOp: : :	 :
 :  : : : : : : 
�B
�
>__inference_lstm_layer_call_and_return_conditional_losses_4015

inputs'
#statefulpartitionedcall_lstm_kernel%
!statefulpartitionedcall_lstm_bias1
-statefulpartitionedcall_lstm_recurrent_kernel
identity��StatefulPartitionedCall�while;
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
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
dtype0*
_output_shapes
: *
value	B :@_
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
value	B :@*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@O
zeros_1/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
_output_shapes
: *
T0R
zeros_1/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
_output_shapes
:*
T0R
zeros_1/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@c
transpose/permConst*
dtype0*
_output_shapes
:*!
valueB"          v
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
strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:a
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
T0�
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0#statefulpartitionedcall_lstm_kernel!statefulpartitionedcall_lstm_bias-statefulpartitionedcall_lstm_recurrent_kernel*L
fGRE
C__inference_lstm_cell_layer_call_and_return_conditional_losses_3501*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin

2*M
_output_shapes;
9:���������@:���������@:���������@*+
_gradient_op_typePartitionedCall-3525n
TensorArrayV2_1/element_shapeConst*
valueB"����@   *
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#statefulpartitionedcall_lstm_kernel!statefulpartitionedcall_lstm_bias-statefulpartitionedcall_lstm_recurrent_kernel^StatefulPartitionedCall*
T
2*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_3938*
_num_original_outputs*
bodyR
while_body_3939*L
_output_shapes:
8: : : : :���������@:���������@: : : : : K
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
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:���������@^
while/Identity_5Identitywhile:output:5*'
_output_shapes
:���������@*
T0M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
_output_shapes
: *
T0M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
_output_shapes
: *
T0�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"����@   *
dtype0*
_output_shapes
:�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :������������������@h
strided_slice_3/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:���������@*
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
 :������������������@�
IdentityIdentitytranspose_1:y:0^StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :������������������@"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile:& "
 
_user_specified_nameinputs: : : 
�
�
while_cond_3938
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
lstm_kernel
	lstm_bias
lstm_recurrent_kernel
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
identityIdentity:output:0*Q
_input_shapes@
>: : : : :���������@:���������@: : ::::  : : : : : : : : :	 :
 
�
�
while_cond_8099
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
lstm_kernel
	lstm_bias
lstm_recurrent_kernel
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
_output_shapes
: *
T0
"
identityIdentity:output:0*Q
_input_shapes@
>: : : : :���������@:���������@: : :::: : : :	 :
 :  : : : : : 
�
�
while_cond_9247
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
lstm_1_kernel
lstm_1_bias
lstm_1_recurrent_kernel
identity
P
LessLessplaceholderless_strided_slice_1*
_output_shapes
: *
T0?
IdentityIdentityLess:z:0*
_output_shapes
: *
T0
"
identityIdentity:output:0*Q
_input_shapes@
>: : : : :��������� :��������� : : ::::
 :  : : : : : : : : :	 
�`
�
while_body_9248
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0(
$split_readvariableop_lstm_1_kernel_0(
$split_1_readvariableop_lstm_1_bias_0,
(readvariableop_lstm_1_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor&
"split_readvariableop_lstm_1_kernel&
"split_1_readvariableop_lstm_1_bias*
&readvariableop_lstm_1_recurrent_kernel��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����@   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:���������@G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: z
split/ReadVariableOpReadVariableOp$split_readvariableop_lstm_1_kernel_0*
dtype0*
_output_shapes
:	@��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split*<
_output_shapes*
(:@ :@ :@ :@ ~
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:0*
T0*'
_output_shapes
:��������� �
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:1*
T0*'
_output_shapes
:��������� �
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:2*
T0*'
_output_shapes
:��������� �
MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:3*
T0*'
_output_shapes
:��������� I
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: x
split_1/ReadVariableOpReadVariableOp$split_1_readvariableop_lstm_1_bias_0*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
	num_split*,
_output_shapes
: : : : *
T0h
BiasAddBiasAddMatMul:product:0split_1:output:0*'
_output_shapes
:��������� *
T0l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*'
_output_shapes
:��������� *
T0l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:��������� l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:��������� x
ReadVariableOpReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0*
_output_shapes
:	 �*
dtype0d
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

:  *
T0*
Index0k
MatMul_4MatMulplaceholder_2strided_slice:output:0*'
_output_shapes
:��������� *
T0d
addAddV2BiasAdd:output:0MatMul_4:product:0*'
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
: W
MulMuladd:z:0Const_2:output:0*'
_output_shapes
:��������� *
T0Y
Add_1AddMul:z:0Const_3:output:0*'
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
clip_by_value/yConst*
_output_shapes
: *
valueB
 *    *
dtype0
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_1ReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0^ReadVariableOp*
dtype0*
_output_shapes
:	 �f
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
strided_slice_2StridedSliceReadVariableOp_1:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:  m
MatMul_5MatMulplaceholder_2strided_slice_2:output:0*
T0*'
_output_shapes
:��������� h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:��������� L
Const_4Const*
dtype0*
_output_shapes
: *
valueB
 *��L>L
Const_5Const*
_output_shapes
: *
valueB
 *   ?*
dtype0[
Mul_1Mul	add_2:z:0Const_4:output:0*'
_output_shapes
:��������� *
T0[
Add_3Add	Mul_1:z:0Const_5:output:0*
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
mul_2Mulclip_by_value_1:z:0placeholder_3*
T0*'
_output_shapes
:��������� �
ReadVariableOp_2ReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0^ReadVariableOp_1*
dtype0*
_output_shapes
:	 �f
strided_slice_3/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_3/stack_1Const*
_output_shapes
:*
valueB"    `   *
dtype0h
strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
strided_slice_3StridedSliceReadVariableOp_2:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
Index0*
T0m
MatMul_6MatMulplaceholder_2strided_slice_3:output:0*
T0*'
_output_shapes
:��������� h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*'
_output_shapes
:��������� *
T0I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:��������� [
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_5AddV2	mul_2:z:0	mul_3:z:0*'
_output_shapes
:��������� *
T0�
ReadVariableOp_3ReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0^ReadVariableOp_2*
_output_shapes
:	 �*
dtype0f
strided_slice_4/stackConst*
valueB"    `   *
dtype0*
_output_shapes
:h
strided_slice_4/stack_1Const*
dtype0*
_output_shapes
:*
valueB"        h
strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_4StridedSliceReadVariableOp_3:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
Index0*
T0m
MatMul_7MatMulplaceholder_2strided_slice_4:output:0*'
_output_shapes
:��������� *
T0h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:��������� L
Const_6Const*
dtype0*
_output_shapes
: *
valueB
 *��L>L
Const_7Const*
dtype0*
_output_shapes
: *
valueB
 *   ?[
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:��������� [
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:��������� ^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
valueB
 *  �?*
dtype0�
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*'
_output_shapes
:��������� *
T0V
clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*'
_output_shapes
:��������� *
T0K
Tanh_1Tanh	add_5:z:0*'
_output_shapes
:��������� *
T0_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*'
_output_shapes
:��������� *
T0�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_5:z:0*
element_dtype0*
_output_shapes
: I
add_8/yConst*
dtype0*
_output_shapes
: *
value	B :N
add_8AddV2placeholderadd_8/y:output:0*
T0*
_output_shapes
: I
add_9/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_9AddV2while_loop_counteradd_9/y:output:0*
_output_shapes
: *
T0�
IdentityIdentity	add_9:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_1Identitywhile_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_2Identity	add_8:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_4Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:��������� �

Identity_5Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:��������� "J
"split_readvariableop_lstm_1_kernel$split_readvariableop_lstm_1_kernel_0"R
&readvariableop_lstm_1_recurrent_kernel(readvariableop_lstm_1_recurrent_kernel_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"J
"split_1_readvariableop_lstm_1_bias$split_1_readvariableop_lstm_1_bias_0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"$
strided_slice_1strided_slice_1_0*Q
_input_shapes@
>: : : : :��������� :��������� : : :::2 
ReadVariableOpReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp:	 :
 :  : : : : : : : : 
�
�
B__inference_features_layer_call_and_return_conditional_losses_4781

inputs(
$embedding_lookup_features_embeddings
identity��embedding_lookupU
CastCastinputs*

SrcT0*

DstT0*'
_output_shapes
:���������(�
embedding_lookupResourceGather$embedding_lookup_features_embeddingsCast:y:0*
Tindices0*
dtype0*+
_output_shapes
:���������(*7
_class-
+)loc:@embedding_lookup/features/embeddings�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*7
_class-
+)loc:@embedding_lookup/features/embeddings*+
_output_shapes
:���������(�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*+
_output_shapes
:���������(*
T0�
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:���������("
identityIdentity:output:0**
_input_shapes
:���������(:2$
embedding_lookupembedding_lookup:& "
 
_user_specified_nameinputs: 
�
�
)__inference_sequential_layer_call_fn_6128
features_input/
+statefulpartitionedcall_features_embeddings'
#statefulpartitionedcall_lstm_kernel%
!statefulpartitionedcall_lstm_bias1
-statefulpartitionedcall_lstm_recurrent_kernel)
%statefulpartitionedcall_lstm_1_kernel'
#statefulpartitionedcall_lstm_1_bias3
/statefulpartitionedcall_lstm_1_recurrent_kernel(
$statefulpartitionedcall_dense_kernel&
"statefulpartitionedcall_dense_bias*
&statefulpartitionedcall_dense_1_kernel(
$statefulpartitionedcall_dense_1_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallfeatures_input+statefulpartitionedcall_features_embeddings#statefulpartitionedcall_lstm_kernel!statefulpartitionedcall_lstm_bias-statefulpartitionedcall_lstm_recurrent_kernel%statefulpartitionedcall_lstm_1_kernel#statefulpartitionedcall_lstm_1_bias/statefulpartitionedcall_lstm_1_recurrent_kernel$statefulpartitionedcall_dense_kernel"statefulpartitionedcall_dense_bias&statefulpartitionedcall_dense_1_kernel$statefulpartitionedcall_dense_1_bias*+
_gradient_op_typePartitionedCall-6114*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_6113*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:���������*
Tin
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*R
_input_shapesA
?:���������(:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:. *
(
_user_specified_namefeatures_input: : : : : : : : :	 :
 : 
�a
�
lstm_1_while_body_7185
lstm_1_while_loop_counter#
lstm_1_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
lstm_1_strided_slice_1_0X
Ttensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0(
$split_readvariableop_lstm_1_kernel_0(
$split_1_readvariableop_lstm_1_bias_0,
(readvariableop_lstm_1_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
lstm_1_strided_slice_1V
Rtensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor&
"split_readvariableop_lstm_1_kernel&
"split_1_readvariableop_lstm_1_bias*
&readvariableop_lstm_1_recurrent_kernel��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����@   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemTtensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:���������@G
ConstConst*
dtype0*
_output_shapes
: *
value	B :Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: z
split/ReadVariableOpReadVariableOp$split_readvariableop_lstm_1_kernel_0*
dtype0*
_output_shapes
:	@��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split*<
_output_shapes*
(:@ :@ :@ :@ ~
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:0*'
_output_shapes
:��������� *
T0�
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:1*
T0*'
_output_shapes
:��������� �
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:2*
T0*'
_output_shapes
:��������� �
MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:3*'
_output_shapes
:��������� *
T0I
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: x
split_1/ReadVariableOpReadVariableOp$split_1_readvariableop_lstm_1_bias_0*
_output_shapes	
:�*
dtype0�
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
	num_split*,
_output_shapes
: : : : *
T0h
BiasAddBiasAddMatMul:product:0split_1:output:0*'
_output_shapes
:��������� *
T0l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:��������� l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:��������� l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:��������� x
ReadVariableOpReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0*
dtype0*
_output_shapes
:	 �d
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
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
_output_shapes

:  *
Index0*
T0*

begin_mask*
end_maskk
MatMul_4MatMulplaceholder_2strided_slice:output:0*
T0*'
_output_shapes
:��������� d
addAddV2BiasAdd:output:0MatMul_4:product:0*
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
: W
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:��������� Y
Add_1AddMul:z:0Const_3:output:0*
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
:��������� �
ReadVariableOp_1ReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0^ReadVariableOp*
dtype0*
_output_shapes
:	 �f
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

:  *
Index0*
T0m
MatMul_5MatMulplaceholder_2strided_slice_1:output:0*
T0*'
_output_shapes
:��������� h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*'
_output_shapes
:��������� *
T0L
Const_4Const*
_output_shapes
: *
valueB
 *��L>*
dtype0L
Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:��������� [
Add_3Add	Mul_1:z:0Const_5:output:0*'
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
mul_2Mulclip_by_value_1:z:0placeholder_3*
T0*'
_output_shapes
:��������� �
ReadVariableOp_2ReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0^ReadVariableOp_1*
dtype0*
_output_shapes
:	 �f
strided_slice_2/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB"    `   h
strided_slice_2/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
Index0*
T0m
MatMul_6MatMulplaceholder_2strided_slice_2:output:0*'
_output_shapes
:��������� *
T0h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_4:z:0*'
_output_shapes
:��������� *
T0[
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_3ReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0^ReadVariableOp_2*
dtype0*
_output_shapes
:	 �f
strided_slice_3/stackConst*
valueB"    `   *
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
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
end_mask*
_output_shapes

:  *
T0*
Index0*

begin_maskm
MatMul_7MatMulplaceholder_2strided_slice_3:output:0*
T0*'
_output_shapes
:��������� h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*'
_output_shapes
:��������� *
T0L
Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_4Mul	add_6:z:0Const_6:output:0*'
_output_shapes
:��������� *
T0[
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:��������� ^
clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:��������� V
clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:��������� K
Tanh_1Tanh	add_5:z:0*'
_output_shapes
:��������� *
T0_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*'
_output_shapes
:��������� *
T0�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_5:z:0*
element_dtype0*
_output_shapes
: I
add_8/yConst*
_output_shapes
: *
value	B :*
dtype0N
add_8AddV2placeholderadd_8/y:output:0*
_output_shapes
: *
T0I
add_9/yConst*
value	B :*
dtype0*
_output_shapes
: \
add_9AddV2lstm_1_while_loop_counteradd_9/y:output:0*
_output_shapes
: *
T0�
IdentityIdentity	add_9:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_1Identitylstm_1_while_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_2Identity	add_8:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_4Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*'
_output_shapes
:��������� *
T0�

Identity_5Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*'
_output_shapes
:��������� *
T0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"J
"split_readvariableop_lstm_1_kernel$split_readvariableop_lstm_1_kernel_0"�
Rtensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensorTtensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0"R
&readvariableop_lstm_1_recurrent_kernel(readvariableop_lstm_1_recurrent_kernel_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"J
"split_1_readvariableop_lstm_1_bias$split_1_readvariableop_lstm_1_bias_0"2
lstm_1_strided_slice_1lstm_1_strided_slice_1_0"!

identity_4Identity_4:output:0*Q
_input_shapes@
>: : : : :��������� :��������� : : :::2 
ReadVariableOpReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp:  : : : : : : : : :	 :
 
��
�
__inference__wrapped_model_3273
features_input<
8sequential_features_embedding_lookup_features_embeddings4
0sequential_lstm_split_readvariableop_lstm_kernel4
0sequential_lstm_split_1_readvariableop_lstm_bias8
4sequential_lstm_readvariableop_lstm_recurrent_kernel8
4sequential_lstm_1_split_readvariableop_lstm_1_kernel8
4sequential_lstm_1_split_1_readvariableop_lstm_1_bias<
8sequential_lstm_1_readvariableop_lstm_1_recurrent_kernel7
3sequential_dense_matmul_readvariableop_dense_kernel6
2sequential_dense_biasadd_readvariableop_dense_bias;
7sequential_dense_1_matmul_readvariableop_dense_1_kernel:
6sequential_dense_1_biasadd_readvariableop_dense_1_bias
identity��'sequential/dense/BiasAdd/ReadVariableOp�&sequential/dense/MatMul/ReadVariableOp�)sequential/dense_1/BiasAdd/ReadVariableOp�(sequential/dense_1/MatMul/ReadVariableOp�$sequential/features/embedding_lookup�sequential/lstm/ReadVariableOp� sequential/lstm/ReadVariableOp_1� sequential/lstm/ReadVariableOp_2� sequential/lstm/ReadVariableOp_3�$sequential/lstm/split/ReadVariableOp�&sequential/lstm/split_1/ReadVariableOp�sequential/lstm/while� sequential/lstm_1/ReadVariableOp�"sequential/lstm_1/ReadVariableOp_1�"sequential/lstm_1/ReadVariableOp_2�"sequential/lstm_1/ReadVariableOp_3�&sequential/lstm_1/split/ReadVariableOp�(sequential/lstm_1/split_1/ReadVariableOp�sequential/lstm_1/whileq
sequential/features/CastCastfeatures_input*

SrcT0*

DstT0*'
_output_shapes
:���������(�
$sequential/features/embedding_lookupResourceGather8sequential_features_embedding_lookup_features_embeddingssequential/features/Cast:y:0*+
_output_shapes
:���������(*K
_classA
?=loc:@sequential/features/embedding_lookup/features/embeddings*
Tindices0*
dtype0�
-sequential/features/embedding_lookup/IdentityIdentity-sequential/features/embedding_lookup:output:0*
T0*K
_classA
?=loc:@sequential/features/embedding_lookup/features/embeddings*+
_output_shapes
:���������(�
/sequential/features/embedding_lookup/Identity_1Identity6sequential/features/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������(}
sequential/lstm/ShapeShape8sequential/features/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:m
#sequential/lstm/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:o
%sequential/lstm/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:o
%sequential/lstm/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
sequential/lstm/strided_sliceStridedSlicesequential/lstm/Shape:output:0,sequential/lstm/strided_slice/stack:output:0.sequential/lstm/strided_slice/stack_1:output:0.sequential/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: ]
sequential/lstm/zeros/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: �
sequential/lstm/zeros/mulMul&sequential/lstm/strided_slice:output:0$sequential/lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: _
sequential/lstm/zeros/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: �
sequential/lstm/zeros/LessLesssequential/lstm/zeros/mul:z:0%sequential/lstm/zeros/Less/y:output:0*
_output_shapes
: *
T0`
sequential/lstm/zeros/packed/1Const*
_output_shapes
: *
value	B :@*
dtype0�
sequential/lstm/zeros/packedPack&sequential/lstm/strided_slice:output:0'sequential/lstm/zeros/packed/1:output:0*
T0*
N*
_output_shapes
:`
sequential/lstm/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: �
sequential/lstm/zerosFill%sequential/lstm/zeros/packed:output:0$sequential/lstm/zeros/Const:output:0*
T0*'
_output_shapes
:���������@_
sequential/lstm/zeros_1/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: �
sequential/lstm/zeros_1/mulMul&sequential/lstm/strided_slice:output:0&sequential/lstm/zeros_1/mul/y:output:0*
_output_shapes
: *
T0a
sequential/lstm/zeros_1/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: �
sequential/lstm/zeros_1/LessLesssequential/lstm/zeros_1/mul:z:0'sequential/lstm/zeros_1/Less/y:output:0*
_output_shapes
: *
T0b
 sequential/lstm/zeros_1/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: �
sequential/lstm/zeros_1/packedPack&sequential/lstm/strided_slice:output:0)sequential/lstm/zeros_1/packed/1:output:0*
_output_shapes
:*
T0*
Nb
sequential/lstm/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: �
sequential/lstm/zeros_1Fill'sequential/lstm/zeros_1/packed:output:0&sequential/lstm/zeros_1/Const:output:0*'
_output_shapes
:���������@*
T0s
sequential/lstm/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
sequential/lstm/transpose	Transpose8sequential/features/embedding_lookup/Identity_1:output:0'sequential/lstm/transpose/perm:output:0*
T0*+
_output_shapes
:(���������d
sequential/lstm/Shape_1Shapesequential/lstm/transpose:y:0*
_output_shapes
:*
T0o
%sequential/lstm/strided_slice_1/stackConst*
_output_shapes
:*
valueB: *
dtype0q
'sequential/lstm/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:q
'sequential/lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
sequential/lstm/strided_slice_1StridedSlice sequential/lstm/Shape_1:output:0.sequential/lstm/strided_slice_1/stack:output:00sequential/lstm/strided_slice_1/stack_1:output:00sequential/lstm/strided_slice_1/stack_2:output:0*
_output_shapes
: *
T0*
Index0*
shrink_axis_maskv
+sequential/lstm/TensorArrayV2/element_shapeConst*
dtype0*
_output_shapes
: *
valueB :
����������
sequential/lstm/TensorArrayV2TensorListReserve4sequential/lstm/TensorArrayV2/element_shape:output:0(sequential/lstm/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: �
Esequential/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
7sequential/lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsequential/lstm/transpose:y:0Nsequential/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: o
%sequential/lstm/strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:q
'sequential/lstm/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:q
'sequential/lstm/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
sequential/lstm/strided_slice_2StridedSlicesequential/lstm/transpose:y:0.sequential/lstm/strided_slice_2/stack:output:00sequential/lstm/strided_slice_2/stack_1:output:00sequential/lstm/strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:���������*
T0*
Index0W
sequential/lstm/ConstConst*
_output_shapes
: *
value	B :*
dtype0a
sequential/lstm/split/split_dimConst*
_output_shapes
: *
value	B :*
dtype0�
$sequential/lstm/split/ReadVariableOpReadVariableOp0sequential_lstm_split_readvariableop_lstm_kernel*
dtype0*
_output_shapes
:	��
sequential/lstm/splitSplit(sequential/lstm/split/split_dim:output:0,sequential/lstm/split/ReadVariableOp:value:0*
T0*
	num_split*<
_output_shapes*
(:@:@:@:@�
sequential/lstm/MatMulMatMul(sequential/lstm/strided_slice_2:output:0sequential/lstm/split:output:0*
T0*'
_output_shapes
:���������@�
sequential/lstm/MatMul_1MatMul(sequential/lstm/strided_slice_2:output:0sequential/lstm/split:output:1*
T0*'
_output_shapes
:���������@�
sequential/lstm/MatMul_2MatMul(sequential/lstm/strided_slice_2:output:0sequential/lstm/split:output:2*'
_output_shapes
:���������@*
T0�
sequential/lstm/MatMul_3MatMul(sequential/lstm/strided_slice_2:output:0sequential/lstm/split:output:3*
T0*'
_output_shapes
:���������@Y
sequential/lstm/Const_1Const*
dtype0*
_output_shapes
: *
value	B :c
!sequential/lstm/split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: �
&sequential/lstm/split_1/ReadVariableOpReadVariableOp0sequential_lstm_split_1_readvariableop_lstm_bias*
_output_shapes	
:�*
dtype0�
sequential/lstm/split_1Split*sequential/lstm/split_1/split_dim:output:0.sequential/lstm/split_1/ReadVariableOp:value:0*
	num_split*,
_output_shapes
:@:@:@:@*
T0�
sequential/lstm/BiasAddBiasAdd sequential/lstm/MatMul:product:0 sequential/lstm/split_1:output:0*
T0*'
_output_shapes
:���������@�
sequential/lstm/BiasAdd_1BiasAdd"sequential/lstm/MatMul_1:product:0 sequential/lstm/split_1:output:1*'
_output_shapes
:���������@*
T0�
sequential/lstm/BiasAdd_2BiasAdd"sequential/lstm/MatMul_2:product:0 sequential/lstm/split_1:output:2*
T0*'
_output_shapes
:���������@�
sequential/lstm/BiasAdd_3BiasAdd"sequential/lstm/MatMul_3:product:0 sequential/lstm/split_1:output:3*
T0*'
_output_shapes
:���������@�
sequential/lstm/ReadVariableOpReadVariableOp4sequential_lstm_readvariableop_lstm_recurrent_kernel*
dtype0*
_output_shapes
:	@�v
%sequential/lstm/strided_slice_3/stackConst*
_output_shapes
:*
valueB"        *
dtype0x
'sequential/lstm/strided_slice_3/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:x
'sequential/lstm/strided_slice_3/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
sequential/lstm/strided_slice_3StridedSlice&sequential/lstm/ReadVariableOp:value:0.sequential/lstm/strided_slice_3/stack:output:00sequential/lstm/strided_slice_3/stack_1:output:00sequential/lstm/strided_slice_3/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:@@*
Index0*
T0�
sequential/lstm/MatMul_4MatMulsequential/lstm/zeros:output:0(sequential/lstm/strided_slice_3:output:0*'
_output_shapes
:���������@*
T0�
sequential/lstm/addAddV2 sequential/lstm/BiasAdd:output:0"sequential/lstm/MatMul_4:product:0*'
_output_shapes
:���������@*
T0\
sequential/lstm/Const_2Const*
valueB
 *��L>*
dtype0*
_output_shapes
: \
sequential/lstm/Const_3Const*
valueB
 *   ?*
dtype0*
_output_shapes
: �
sequential/lstm/MulMulsequential/lstm/add:z:0 sequential/lstm/Const_2:output:0*
T0*'
_output_shapes
:���������@�
sequential/lstm/Add_1Addsequential/lstm/Mul:z:0 sequential/lstm/Const_3:output:0*
T0*'
_output_shapes
:���������@l
'sequential/lstm/clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
%sequential/lstm/clip_by_value/MinimumMinimumsequential/lstm/Add_1:z:00sequential/lstm/clip_by_value/Minimum/y:output:0*'
_output_shapes
:���������@*
T0d
sequential/lstm/clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
sequential/lstm/clip_by_valueMaximum)sequential/lstm/clip_by_value/Minimum:z:0(sequential/lstm/clip_by_value/y:output:0*
T0*'
_output_shapes
:���������@�
 sequential/lstm/ReadVariableOp_1ReadVariableOp4sequential_lstm_readvariableop_lstm_recurrent_kernel^sequential/lstm/ReadVariableOp*
dtype0*
_output_shapes
:	@�v
%sequential/lstm/strided_slice_4/stackConst*
_output_shapes
:*
valueB"    @   *
dtype0x
'sequential/lstm/strided_slice_4/stack_1Const*
dtype0*
_output_shapes
:*
valueB"    �   x
'sequential/lstm/strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
sequential/lstm/strided_slice_4StridedSlice(sequential/lstm/ReadVariableOp_1:value:0.sequential/lstm/strided_slice_4/stack:output:00sequential/lstm/strided_slice_4/stack_1:output:00sequential/lstm/strided_slice_4/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:@@�
sequential/lstm/MatMul_5MatMulsequential/lstm/zeros:output:0(sequential/lstm/strided_slice_4:output:0*
T0*'
_output_shapes
:���������@�
sequential/lstm/add_2AddV2"sequential/lstm/BiasAdd_1:output:0"sequential/lstm/MatMul_5:product:0*
T0*'
_output_shapes
:���������@\
sequential/lstm/Const_4Const*
valueB
 *��L>*
dtype0*
_output_shapes
: \
sequential/lstm/Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: �
sequential/lstm/Mul_1Mulsequential/lstm/add_2:z:0 sequential/lstm/Const_4:output:0*
T0*'
_output_shapes
:���������@�
sequential/lstm/Add_3Addsequential/lstm/Mul_1:z:0 sequential/lstm/Const_5:output:0*'
_output_shapes
:���������@*
T0n
)sequential/lstm/clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
'sequential/lstm/clip_by_value_1/MinimumMinimumsequential/lstm/Add_3:z:02sequential/lstm/clip_by_value_1/Minimum/y:output:0*'
_output_shapes
:���������@*
T0f
!sequential/lstm/clip_by_value_1/yConst*
_output_shapes
: *
valueB
 *    *
dtype0�
sequential/lstm/clip_by_value_1Maximum+sequential/lstm/clip_by_value_1/Minimum:z:0*sequential/lstm/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:���������@�
sequential/lstm/mul_2Mul#sequential/lstm/clip_by_value_1:z:0 sequential/lstm/zeros_1:output:0*'
_output_shapes
:���������@*
T0�
 sequential/lstm/ReadVariableOp_2ReadVariableOp4sequential_lstm_readvariableop_lstm_recurrent_kernel!^sequential/lstm/ReadVariableOp_1*
dtype0*
_output_shapes
:	@�v
%sequential/lstm/strided_slice_5/stackConst*
valueB"    �   *
dtype0*
_output_shapes
:x
'sequential/lstm/strided_slice_5/stack_1Const*
valueB"    �   *
dtype0*
_output_shapes
:x
'sequential/lstm/strided_slice_5/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
sequential/lstm/strided_slice_5StridedSlice(sequential/lstm/ReadVariableOp_2:value:0.sequential/lstm/strided_slice_5/stack:output:00sequential/lstm/strided_slice_5/stack_1:output:00sequential/lstm/strided_slice_5/stack_2:output:0*
end_mask*
_output_shapes

:@@*
T0*
Index0*

begin_mask�
sequential/lstm/MatMul_6MatMulsequential/lstm/zeros:output:0(sequential/lstm/strided_slice_5:output:0*
T0*'
_output_shapes
:���������@�
sequential/lstm/add_4AddV2"sequential/lstm/BiasAdd_2:output:0"sequential/lstm/MatMul_6:product:0*
T0*'
_output_shapes
:���������@i
sequential/lstm/TanhTanhsequential/lstm/add_4:z:0*
T0*'
_output_shapes
:���������@�
sequential/lstm/mul_3Mul!sequential/lstm/clip_by_value:z:0sequential/lstm/Tanh:y:0*'
_output_shapes
:���������@*
T0�
sequential/lstm/add_5AddV2sequential/lstm/mul_2:z:0sequential/lstm/mul_3:z:0*'
_output_shapes
:���������@*
T0�
 sequential/lstm/ReadVariableOp_3ReadVariableOp4sequential_lstm_readvariableop_lstm_recurrent_kernel!^sequential/lstm/ReadVariableOp_2*
dtype0*
_output_shapes
:	@�v
%sequential/lstm/strided_slice_6/stackConst*
valueB"    �   *
dtype0*
_output_shapes
:x
'sequential/lstm/strided_slice_6/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:x
'sequential/lstm/strided_slice_6/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
sequential/lstm/strided_slice_6StridedSlice(sequential/lstm/ReadVariableOp_3:value:0.sequential/lstm/strided_slice_6/stack:output:00sequential/lstm/strided_slice_6/stack_1:output:00sequential/lstm/strided_slice_6/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:@@*
T0*
Index0�
sequential/lstm/MatMul_7MatMulsequential/lstm/zeros:output:0(sequential/lstm/strided_slice_6:output:0*
T0*'
_output_shapes
:���������@�
sequential/lstm/add_6AddV2"sequential/lstm/BiasAdd_3:output:0"sequential/lstm/MatMul_7:product:0*
T0*'
_output_shapes
:���������@\
sequential/lstm/Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: \
sequential/lstm/Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: �
sequential/lstm/Mul_4Mulsequential/lstm/add_6:z:0 sequential/lstm/Const_6:output:0*'
_output_shapes
:���������@*
T0�
sequential/lstm/Add_7Addsequential/lstm/Mul_4:z:0 sequential/lstm/Const_7:output:0*'
_output_shapes
:���������@*
T0n
)sequential/lstm/clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
'sequential/lstm/clip_by_value_2/MinimumMinimumsequential/lstm/Add_7:z:02sequential/lstm/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:���������@f
!sequential/lstm/clip_by_value_2/yConst*
_output_shapes
: *
valueB
 *    *
dtype0�
sequential/lstm/clip_by_value_2Maximum+sequential/lstm/clip_by_value_2/Minimum:z:0*sequential/lstm/clip_by_value_2/y:output:0*'
_output_shapes
:���������@*
T0k
sequential/lstm/Tanh_1Tanhsequential/lstm/add_5:z:0*
T0*'
_output_shapes
:���������@�
sequential/lstm/mul_5Mul#sequential/lstm/clip_by_value_2:z:0sequential/lstm/Tanh_1:y:0*
T0*'
_output_shapes
:���������@~
-sequential/lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
valueB"����@   *
dtype0�
sequential/lstm/TensorArrayV2_1TensorListReserve6sequential/lstm/TensorArrayV2_1/element_shape:output:0(sequential/lstm/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: V
sequential/lstm/timeConst*
value	B : *
dtype0*
_output_shapes
: s
(sequential/lstm/while/maximum_iterationsConst*
_output_shapes
: *
valueB :
���������*
dtype0d
"sequential/lstm/while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: �
sequential/lstm/whileWhile+sequential/lstm/while/loop_counter:output:01sequential/lstm/while/maximum_iterations:output:0sequential/lstm/time:output:0(sequential/lstm/TensorArrayV2_1:handle:0sequential/lstm/zeros:output:0 sequential/lstm/zeros_1:output:0(sequential/lstm/strided_slice_1:output:0Gsequential/lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:00sequential_lstm_split_readvariableop_lstm_kernel0sequential_lstm_split_1_readvariableop_lstm_bias4sequential_lstm_readvariableop_lstm_recurrent_kernel!^sequential/lstm/ReadVariableOp_3%^sequential/lstm/split/ReadVariableOp'^sequential/lstm/split_1/ReadVariableOp*+
body#R!
sequential_lstm_while_body_2830*L
_output_shapes:
8: : : : :���������@:���������@: : : : : *
T
2*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
_lower_using_switch_merge(*
parallel_iterations *+
cond#R!
sequential_lstm_while_cond_2829*
_num_original_outputsk
sequential/lstm/while/IdentityIdentitysequential/lstm/while:output:0*
T0*
_output_shapes
: m
 sequential/lstm/while/Identity_1Identitysequential/lstm/while:output:1*
_output_shapes
: *
T0m
 sequential/lstm/while/Identity_2Identitysequential/lstm/while:output:2*
T0*
_output_shapes
: m
 sequential/lstm/while/Identity_3Identitysequential/lstm/while:output:3*
T0*
_output_shapes
: ~
 sequential/lstm/while/Identity_4Identitysequential/lstm/while:output:4*
T0*'
_output_shapes
:���������@~
 sequential/lstm/while/Identity_5Identitysequential/lstm/while:output:5*
T0*'
_output_shapes
:���������@m
 sequential/lstm/while/Identity_6Identitysequential/lstm/while:output:6*
T0*
_output_shapes
: m
 sequential/lstm/while/Identity_7Identitysequential/lstm/while:output:7*
T0*
_output_shapes
: m
 sequential/lstm/while/Identity_8Identitysequential/lstm/while:output:8*
T0*
_output_shapes
: m
 sequential/lstm/while/Identity_9Identitysequential/lstm/while:output:9*
T0*
_output_shapes
: o
!sequential/lstm/while/Identity_10Identitysequential/lstm/while:output:10*
_output_shapes
: *
T0�
@sequential/lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"����@   *
dtype0*
_output_shapes
:�
2sequential/lstm/TensorArrayV2Stack/TensorListStackTensorListStack)sequential/lstm/while/Identity_3:output:0Isequential/lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:(���������@x
%sequential/lstm/strided_slice_7/stackConst*
_output_shapes
:*
valueB:
���������*
dtype0q
'sequential/lstm/strided_slice_7/stack_1Const*
dtype0*
_output_shapes
:*
valueB: q
'sequential/lstm/strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
sequential/lstm/strided_slice_7StridedSlice;sequential/lstm/TensorArrayV2Stack/TensorListStack:tensor:0.sequential/lstm/strided_slice_7/stack:output:00sequential/lstm/strided_slice_7/stack_1:output:00sequential/lstm/strided_slice_7/stack_2:output:0*'
_output_shapes
:���������@*
Index0*
T0*
shrink_axis_masku
 sequential/lstm/transpose_1/permConst*
_output_shapes
:*!
valueB"          *
dtype0�
sequential/lstm/transpose_1	Transpose;sequential/lstm/TensorArrayV2Stack/TensorListStack:tensor:0)sequential/lstm/transpose_1/perm:output:0*+
_output_shapes
:���������(@*
T0f
sequential/lstm_1/ShapeShapesequential/lstm/transpose_1:y:0*
_output_shapes
:*
T0o
%sequential/lstm_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:q
'sequential/lstm_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:q
'sequential/lstm_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
sequential/lstm_1/strided_sliceStridedSlice sequential/lstm_1/Shape:output:0.sequential/lstm_1/strided_slice/stack:output:00sequential/lstm_1/strided_slice/stack_1:output:00sequential/lstm_1/strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0_
sequential/lstm_1/zeros/mul/yConst*
value	B : *
dtype0*
_output_shapes
: �
sequential/lstm_1/zeros/mulMul(sequential/lstm_1/strided_slice:output:0&sequential/lstm_1/zeros/mul/y:output:0*
T0*
_output_shapes
: a
sequential/lstm_1/zeros/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: �
sequential/lstm_1/zeros/LessLesssequential/lstm_1/zeros/mul:z:0'sequential/lstm_1/zeros/Less/y:output:0*
T0*
_output_shapes
: b
 sequential/lstm_1/zeros/packed/1Const*
value	B : *
dtype0*
_output_shapes
: �
sequential/lstm_1/zeros/packedPack(sequential/lstm_1/strided_slice:output:0)sequential/lstm_1/zeros/packed/1:output:0*
_output_shapes
:*
T0*
Nb
sequential/lstm_1/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0�
sequential/lstm_1/zerosFill'sequential/lstm_1/zeros/packed:output:0&sequential/lstm_1/zeros/Const:output:0*'
_output_shapes
:��������� *
T0a
sequential/lstm_1/zeros_1/mul/yConst*
value	B : *
dtype0*
_output_shapes
: �
sequential/lstm_1/zeros_1/mulMul(sequential/lstm_1/strided_slice:output:0(sequential/lstm_1/zeros_1/mul/y:output:0*
T0*
_output_shapes
: c
 sequential/lstm_1/zeros_1/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: �
sequential/lstm_1/zeros_1/LessLess!sequential/lstm_1/zeros_1/mul:z:0)sequential/lstm_1/zeros_1/Less/y:output:0*
T0*
_output_shapes
: d
"sequential/lstm_1/zeros_1/packed/1Const*
value	B : *
dtype0*
_output_shapes
: �
 sequential/lstm_1/zeros_1/packedPack(sequential/lstm_1/strided_slice:output:0+sequential/lstm_1/zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:d
sequential/lstm_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: �
sequential/lstm_1/zeros_1Fill)sequential/lstm_1/zeros_1/packed:output:0(sequential/lstm_1/zeros_1/Const:output:0*'
_output_shapes
:��������� *
T0u
 sequential/lstm_1/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
sequential/lstm_1/transpose	Transposesequential/lstm/transpose_1:y:0)sequential/lstm_1/transpose/perm:output:0*+
_output_shapes
:(���������@*
T0h
sequential/lstm_1/Shape_1Shapesequential/lstm_1/transpose:y:0*
_output_shapes
:*
T0q
'sequential/lstm_1/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:s
)sequential/lstm_1/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:s
)sequential/lstm_1/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
!sequential/lstm_1/strided_slice_1StridedSlice"sequential/lstm_1/Shape_1:output:00sequential/lstm_1/strided_slice_1/stack:output:02sequential/lstm_1/strided_slice_1/stack_1:output:02sequential/lstm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: x
-sequential/lstm_1/TensorArrayV2/element_shapeConst*
valueB :
���������*
dtype0*
_output_shapes
: �
sequential/lstm_1/TensorArrayV2TensorListReserve6sequential/lstm_1/TensorArrayV2/element_shape:output:0*sequential/lstm_1/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: �
Gsequential/lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"����@   *
dtype0*
_output_shapes
:�
9sequential/lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsequential/lstm_1/transpose:y:0Psequential/lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: q
'sequential/lstm_1/strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:s
)sequential/lstm_1/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:s
)sequential/lstm_1/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
!sequential/lstm_1/strided_slice_2StridedSlicesequential/lstm_1/transpose:y:00sequential/lstm_1/strided_slice_2/stack:output:02sequential/lstm_1/strided_slice_2/stack_1:output:02sequential/lstm_1/strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:���������@*
T0*
Index0Y
sequential/lstm_1/ConstConst*
dtype0*
_output_shapes
: *
value	B :c
!sequential/lstm_1/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: �
&sequential/lstm_1/split/ReadVariableOpReadVariableOp4sequential_lstm_1_split_readvariableop_lstm_1_kernel*
dtype0*
_output_shapes
:	@��
sequential/lstm_1/splitSplit*sequential/lstm_1/split/split_dim:output:0.sequential/lstm_1/split/ReadVariableOp:value:0*
T0*
	num_split*<
_output_shapes*
(:@ :@ :@ :@ �
sequential/lstm_1/MatMulMatMul*sequential/lstm_1/strided_slice_2:output:0 sequential/lstm_1/split:output:0*'
_output_shapes
:��������� *
T0�
sequential/lstm_1/MatMul_1MatMul*sequential/lstm_1/strided_slice_2:output:0 sequential/lstm_1/split:output:1*
T0*'
_output_shapes
:��������� �
sequential/lstm_1/MatMul_2MatMul*sequential/lstm_1/strided_slice_2:output:0 sequential/lstm_1/split:output:2*'
_output_shapes
:��������� *
T0�
sequential/lstm_1/MatMul_3MatMul*sequential/lstm_1/strided_slice_2:output:0 sequential/lstm_1/split:output:3*
T0*'
_output_shapes
:��������� [
sequential/lstm_1/Const_1Const*
value	B :*
dtype0*
_output_shapes
: e
#sequential/lstm_1/split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: �
(sequential/lstm_1/split_1/ReadVariableOpReadVariableOp4sequential_lstm_1_split_1_readvariableop_lstm_1_bias*
dtype0*
_output_shapes	
:��
sequential/lstm_1/split_1Split,sequential/lstm_1/split_1/split_dim:output:00sequential/lstm_1/split_1/ReadVariableOp:value:0*
	num_split*,
_output_shapes
: : : : *
T0�
sequential/lstm_1/BiasAddBiasAdd"sequential/lstm_1/MatMul:product:0"sequential/lstm_1/split_1:output:0*
T0*'
_output_shapes
:��������� �
sequential/lstm_1/BiasAdd_1BiasAdd$sequential/lstm_1/MatMul_1:product:0"sequential/lstm_1/split_1:output:1*
T0*'
_output_shapes
:��������� �
sequential/lstm_1/BiasAdd_2BiasAdd$sequential/lstm_1/MatMul_2:product:0"sequential/lstm_1/split_1:output:2*'
_output_shapes
:��������� *
T0�
sequential/lstm_1/BiasAdd_3BiasAdd$sequential/lstm_1/MatMul_3:product:0"sequential/lstm_1/split_1:output:3*
T0*'
_output_shapes
:��������� �
 sequential/lstm_1/ReadVariableOpReadVariableOp8sequential_lstm_1_readvariableop_lstm_1_recurrent_kernel*
dtype0*
_output_shapes
:	 �x
'sequential/lstm_1/strided_slice_3/stackConst*
dtype0*
_output_shapes
:*
valueB"        z
)sequential/lstm_1/strided_slice_3/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:z
)sequential/lstm_1/strided_slice_3/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
!sequential/lstm_1/strided_slice_3StridedSlice(sequential/lstm_1/ReadVariableOp:value:00sequential/lstm_1/strided_slice_3/stack:output:02sequential/lstm_1/strided_slice_3/stack_1:output:02sequential/lstm_1/strided_slice_3/stack_2:output:0*
_output_shapes

:  *
T0*
Index0*

begin_mask*
end_mask�
sequential/lstm_1/MatMul_4MatMul sequential/lstm_1/zeros:output:0*sequential/lstm_1/strided_slice_3:output:0*
T0*'
_output_shapes
:��������� �
sequential/lstm_1/addAddV2"sequential/lstm_1/BiasAdd:output:0$sequential/lstm_1/MatMul_4:product:0*
T0*'
_output_shapes
:��������� ^
sequential/lstm_1/Const_2Const*
valueB
 *��L>*
dtype0*
_output_shapes
: ^
sequential/lstm_1/Const_3Const*
valueB
 *   ?*
dtype0*
_output_shapes
: �
sequential/lstm_1/MulMulsequential/lstm_1/add:z:0"sequential/lstm_1/Const_2:output:0*
T0*'
_output_shapes
:��������� �
sequential/lstm_1/Add_1Addsequential/lstm_1/Mul:z:0"sequential/lstm_1/Const_3:output:0*'
_output_shapes
:��������� *
T0n
)sequential/lstm_1/clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
'sequential/lstm_1/clip_by_value/MinimumMinimumsequential/lstm_1/Add_1:z:02sequential/lstm_1/clip_by_value/Minimum/y:output:0*'
_output_shapes
:��������� *
T0f
!sequential/lstm_1/clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
sequential/lstm_1/clip_by_valueMaximum+sequential/lstm_1/clip_by_value/Minimum:z:0*sequential/lstm_1/clip_by_value/y:output:0*
T0*'
_output_shapes
:��������� �
"sequential/lstm_1/ReadVariableOp_1ReadVariableOp8sequential_lstm_1_readvariableop_lstm_1_recurrent_kernel!^sequential/lstm_1/ReadVariableOp*
dtype0*
_output_shapes
:	 �x
'sequential/lstm_1/strided_slice_4/stackConst*
valueB"        *
dtype0*
_output_shapes
:z
)sequential/lstm_1/strided_slice_4/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:z
)sequential/lstm_1/strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
!sequential/lstm_1/strided_slice_4StridedSlice*sequential/lstm_1/ReadVariableOp_1:value:00sequential/lstm_1/strided_slice_4/stack:output:02sequential/lstm_1/strided_slice_4/stack_1:output:02sequential/lstm_1/strided_slice_4/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
Index0*
T0�
sequential/lstm_1/MatMul_5MatMul sequential/lstm_1/zeros:output:0*sequential/lstm_1/strided_slice_4:output:0*
T0*'
_output_shapes
:��������� �
sequential/lstm_1/add_2AddV2$sequential/lstm_1/BiasAdd_1:output:0$sequential/lstm_1/MatMul_5:product:0*
T0*'
_output_shapes
:��������� ^
sequential/lstm_1/Const_4Const*
_output_shapes
: *
valueB
 *��L>*
dtype0^
sequential/lstm_1/Const_5Const*
dtype0*
_output_shapes
: *
valueB
 *   ?�
sequential/lstm_1/Mul_1Mulsequential/lstm_1/add_2:z:0"sequential/lstm_1/Const_4:output:0*'
_output_shapes
:��������� *
T0�
sequential/lstm_1/Add_3Addsequential/lstm_1/Mul_1:z:0"sequential/lstm_1/Const_5:output:0*
T0*'
_output_shapes
:��������� p
+sequential/lstm_1/clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
)sequential/lstm_1/clip_by_value_1/MinimumMinimumsequential/lstm_1/Add_3:z:04sequential/lstm_1/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:��������� h
#sequential/lstm_1/clip_by_value_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *    �
!sequential/lstm_1/clip_by_value_1Maximum-sequential/lstm_1/clip_by_value_1/Minimum:z:0,sequential/lstm_1/clip_by_value_1/y:output:0*'
_output_shapes
:��������� *
T0�
sequential/lstm_1/mul_2Mul%sequential/lstm_1/clip_by_value_1:z:0"sequential/lstm_1/zeros_1:output:0*'
_output_shapes
:��������� *
T0�
"sequential/lstm_1/ReadVariableOp_2ReadVariableOp8sequential_lstm_1_readvariableop_lstm_1_recurrent_kernel#^sequential/lstm_1/ReadVariableOp_1*
dtype0*
_output_shapes
:	 �x
'sequential/lstm_1/strided_slice_5/stackConst*
dtype0*
_output_shapes
:*
valueB"    @   z
)sequential/lstm_1/strided_slice_5/stack_1Const*
valueB"    `   *
dtype0*
_output_shapes
:z
)sequential/lstm_1/strided_slice_5/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
!sequential/lstm_1/strided_slice_5StridedSlice*sequential/lstm_1/ReadVariableOp_2:value:00sequential/lstm_1/strided_slice_5/stack:output:02sequential/lstm_1/strided_slice_5/stack_1:output:02sequential/lstm_1/strided_slice_5/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
Index0*
T0�
sequential/lstm_1/MatMul_6MatMul sequential/lstm_1/zeros:output:0*sequential/lstm_1/strided_slice_5:output:0*
T0*'
_output_shapes
:��������� �
sequential/lstm_1/add_4AddV2$sequential/lstm_1/BiasAdd_2:output:0$sequential/lstm_1/MatMul_6:product:0*
T0*'
_output_shapes
:��������� m
sequential/lstm_1/TanhTanhsequential/lstm_1/add_4:z:0*
T0*'
_output_shapes
:��������� �
sequential/lstm_1/mul_3Mul#sequential/lstm_1/clip_by_value:z:0sequential/lstm_1/Tanh:y:0*
T0*'
_output_shapes
:��������� �
sequential/lstm_1/add_5AddV2sequential/lstm_1/mul_2:z:0sequential/lstm_1/mul_3:z:0*
T0*'
_output_shapes
:��������� �
"sequential/lstm_1/ReadVariableOp_3ReadVariableOp8sequential_lstm_1_readvariableop_lstm_1_recurrent_kernel#^sequential/lstm_1/ReadVariableOp_2*
dtype0*
_output_shapes
:	 �x
'sequential/lstm_1/strided_slice_6/stackConst*
valueB"    `   *
dtype0*
_output_shapes
:z
)sequential/lstm_1/strided_slice_6/stack_1Const*
_output_shapes
:*
valueB"        *
dtype0z
)sequential/lstm_1/strided_slice_6/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
!sequential/lstm_1/strided_slice_6StridedSlice*sequential/lstm_1/ReadVariableOp_3:value:00sequential/lstm_1/strided_slice_6/stack:output:02sequential/lstm_1/strided_slice_6/stack_1:output:02sequential/lstm_1/strided_slice_6/stack_2:output:0*
_output_shapes

:  *
T0*
Index0*

begin_mask*
end_mask�
sequential/lstm_1/MatMul_7MatMul sequential/lstm_1/zeros:output:0*sequential/lstm_1/strided_slice_6:output:0*
T0*'
_output_shapes
:��������� �
sequential/lstm_1/add_6AddV2$sequential/lstm_1/BiasAdd_3:output:0$sequential/lstm_1/MatMul_7:product:0*
T0*'
_output_shapes
:��������� ^
sequential/lstm_1/Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: ^
sequential/lstm_1/Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: �
sequential/lstm_1/Mul_4Mulsequential/lstm_1/add_6:z:0"sequential/lstm_1/Const_6:output:0*'
_output_shapes
:��������� *
T0�
sequential/lstm_1/Add_7Addsequential/lstm_1/Mul_4:z:0"sequential/lstm_1/Const_7:output:0*
T0*'
_output_shapes
:��������� p
+sequential/lstm_1/clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
)sequential/lstm_1/clip_by_value_2/MinimumMinimumsequential/lstm_1/Add_7:z:04sequential/lstm_1/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:��������� h
#sequential/lstm_1/clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
!sequential/lstm_1/clip_by_value_2Maximum-sequential/lstm_1/clip_by_value_2/Minimum:z:0,sequential/lstm_1/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:��������� o
sequential/lstm_1/Tanh_1Tanhsequential/lstm_1/add_5:z:0*
T0*'
_output_shapes
:��������� �
sequential/lstm_1/mul_5Mul%sequential/lstm_1/clip_by_value_2:z:0sequential/lstm_1/Tanh_1:y:0*'
_output_shapes
:��������� *
T0�
/sequential/lstm_1/TensorArrayV2_1/element_shapeConst*
valueB"����    *
dtype0*
_output_shapes
:�
!sequential/lstm_1/TensorArrayV2_1TensorListReserve8sequential/lstm_1/TensorArrayV2_1/element_shape:output:0*sequential/lstm_1/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: X
sequential/lstm_1/timeConst*
value	B : *
dtype0*
_output_shapes
: u
*sequential/lstm_1/while/maximum_iterationsConst*
valueB :
���������*
dtype0*
_output_shapes
: f
$sequential/lstm_1/while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: �
sequential/lstm_1/whileWhile-sequential/lstm_1/while/loop_counter:output:03sequential/lstm_1/while/maximum_iterations:output:0sequential/lstm_1/time:output:0*sequential/lstm_1/TensorArrayV2_1:handle:0 sequential/lstm_1/zeros:output:0"sequential/lstm_1/zeros_1:output:0*sequential/lstm_1/strided_slice_1:output:0Isequential/lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:04sequential_lstm_1_split_readvariableop_lstm_1_kernel4sequential_lstm_1_split_1_readvariableop_lstm_1_bias8sequential_lstm_1_readvariableop_lstm_1_recurrent_kernel#^sequential/lstm_1/ReadVariableOp_3'^sequential/lstm_1/split/ReadVariableOp)^sequential/lstm_1/split_1/ReadVariableOp*-
cond%R#
!sequential_lstm_1_while_cond_3104*
_num_original_outputs*-
body%R#
!sequential_lstm_1_while_body_3105*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *
T
2*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
_lower_using_switch_merge(*
parallel_iterations o
 sequential/lstm_1/while/IdentityIdentity sequential/lstm_1/while:output:0*
_output_shapes
: *
T0q
"sequential/lstm_1/while/Identity_1Identity sequential/lstm_1/while:output:1*
T0*
_output_shapes
: q
"sequential/lstm_1/while/Identity_2Identity sequential/lstm_1/while:output:2*
T0*
_output_shapes
: q
"sequential/lstm_1/while/Identity_3Identity sequential/lstm_1/while:output:3*
T0*
_output_shapes
: �
"sequential/lstm_1/while/Identity_4Identity sequential/lstm_1/while:output:4*
T0*'
_output_shapes
:��������� �
"sequential/lstm_1/while/Identity_5Identity sequential/lstm_1/while:output:5*
T0*'
_output_shapes
:��������� q
"sequential/lstm_1/while/Identity_6Identity sequential/lstm_1/while:output:6*
T0*
_output_shapes
: q
"sequential/lstm_1/while/Identity_7Identity sequential/lstm_1/while:output:7*
T0*
_output_shapes
: q
"sequential/lstm_1/while/Identity_8Identity sequential/lstm_1/while:output:8*
_output_shapes
: *
T0q
"sequential/lstm_1/while/Identity_9Identity sequential/lstm_1/while:output:9*
T0*
_output_shapes
: s
#sequential/lstm_1/while/Identity_10Identity!sequential/lstm_1/while:output:10*
T0*
_output_shapes
: �
Bsequential/lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
valueB"����    *
dtype0�
4sequential/lstm_1/TensorArrayV2Stack/TensorListStackTensorListStack+sequential/lstm_1/while/Identity_3:output:0Ksequential/lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:(��������� z
'sequential/lstm_1/strided_slice_7/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:s
)sequential/lstm_1/strided_slice_7/stack_1Const*
dtype0*
_output_shapes
:*
valueB: s
)sequential/lstm_1/strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
!sequential/lstm_1/strided_slice_7StridedSlice=sequential/lstm_1/TensorArrayV2Stack/TensorListStack:tensor:00sequential/lstm_1/strided_slice_7/stack:output:02sequential/lstm_1/strided_slice_7/stack_1:output:02sequential/lstm_1/strided_slice_7/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:��������� *
T0*
Index0w
"sequential/lstm_1/transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
sequential/lstm_1/transpose_1	Transpose=sequential/lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0+sequential/lstm_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������( �
&sequential/dense/MatMul/ReadVariableOpReadVariableOp3sequential_dense_matmul_readvariableop_dense_kernel*
dtype0*
_output_shapes

: �
sequential/dense/MatMulMatMul*sequential/lstm_1/strided_slice_7:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_biasadd_readvariableop_dense_bias*
dtype0*
_output_shapes
:�
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:���������*
T0r
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*'
_output_shapes
:���������*
T0~
sequential/dropout/IdentityIdentity#sequential/dense/Relu:activations:0*'
_output_shapes
:���������*
T0�
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp7sequential_dense_1_matmul_readvariableop_dense_1_kernel*
dtype0*
_output_shapes

:�
sequential/dense_1/MatMulMatMul$sequential/dropout/Identity:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp6sequential_dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes
:*
dtype0�
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������|
sequential/dense_1/SigmoidSigmoid#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentitysequential/dense_1/Sigmoid:y:0(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp%^sequential/features/embedding_lookup^sequential/lstm/ReadVariableOp!^sequential/lstm/ReadVariableOp_1!^sequential/lstm/ReadVariableOp_2!^sequential/lstm/ReadVariableOp_3%^sequential/lstm/split/ReadVariableOp'^sequential/lstm/split_1/ReadVariableOp^sequential/lstm/while!^sequential/lstm_1/ReadVariableOp#^sequential/lstm_1/ReadVariableOp_1#^sequential/lstm_1/ReadVariableOp_2#^sequential/lstm_1/ReadVariableOp_3'^sequential/lstm_1/split/ReadVariableOp)^sequential/lstm_1/split_1/ReadVariableOp^sequential/lstm_1/while*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*R
_input_shapesA
?:���������(:::::::::::2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2L
$sequential/lstm/split/ReadVariableOp$sequential/lstm/split/ReadVariableOp2T
(sequential/lstm_1/split_1/ReadVariableOp(sequential/lstm_1/split_1/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/lstm_1/split/ReadVariableOp&sequential/lstm_1/split/ReadVariableOp2D
 sequential/lstm/ReadVariableOp_1 sequential/lstm/ReadVariableOp_12D
 sequential/lstm/ReadVariableOp_2 sequential/lstm/ReadVariableOp_22D
 sequential/lstm/ReadVariableOp_3 sequential/lstm/ReadVariableOp_32L
$sequential/features/embedding_lookup$sequential/features/embedding_lookup2@
sequential/lstm/ReadVariableOpsequential/lstm/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2P
&sequential/lstm/split_1/ReadVariableOp&sequential/lstm/split_1/ReadVariableOp22
sequential/lstm_1/whilesequential/lstm_1/while2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2H
"sequential/lstm_1/ReadVariableOp_1"sequential/lstm_1/ReadVariableOp_12.
sequential/lstm/whilesequential/lstm/while2H
"sequential/lstm_1/ReadVariableOp_2"sequential/lstm_1/ReadVariableOp_22H
"sequential/lstm_1/ReadVariableOp_3"sequential/lstm_1/ReadVariableOp_32D
 sequential/lstm_1/ReadVariableOp sequential/lstm_1/ReadVariableOp:. *
(
_user_specified_namefeatures_input: : : : : : : : :	 :
 : 
�a
�
sequential_lstm_while_body_2830&
"sequential_lstm_while_loop_counter,
(sequential_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3%
!sequential_lstm_strided_slice_1_0a
]tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0&
"split_readvariableop_lstm_kernel_0&
"split_1_readvariableop_lstm_bias_0*
&readvariableop_lstm_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5#
sequential_lstm_strided_slice_1_
[tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor$
 split_readvariableop_lstm_kernel$
 split_1_readvariableop_lstm_bias(
$readvariableop_lstm_recurrent_kernel��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItem]tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:���������G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: x
split/ReadVariableOpReadVariableOp"split_readvariableop_lstm_kernel_0*
dtype0*
_output_shapes
:	��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
	num_split*<
_output_shapes*
(:@:@:@:@*
T0~
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:0*
T0*'
_output_shapes
:���������@�
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:1*
T0*'
_output_shapes
:���������@�
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:2*
T0*'
_output_shapes
:���������@�
MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:3*'
_output_shapes
:���������@*
T0I
Const_1Const*
_output_shapes
: *
value	B :*
dtype0S
split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: v
split_1/ReadVariableOpReadVariableOp"split_1_readvariableop_lstm_bias_0*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split*,
_output_shapes
:@:@:@:@h
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:���������@l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*'
_output_shapes
:���������@*
T0l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:���������@l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:���������@v
ReadVariableOpReadVariableOp&readvariableop_lstm_recurrent_kernel_0*
dtype0*
_output_shapes
:	@�d
strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:f
strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
_output_shapes

:@@*
Index0*
T0*

begin_mask*
end_maskk
MatMul_4MatMulplaceholder_2strided_slice:output:0*
T0*'
_output_shapes
:���������@d
addAddV2BiasAdd:output:0MatMul_4:product:0*'
_output_shapes
:���������@*
T0L
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
 *   ?W
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:���������@Y
Add_1AddMul:z:0Const_3:output:0*'
_output_shapes
:���������@*
T0\
clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������@T
clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������@�
ReadVariableOp_1ReadVariableOp&readvariableop_lstm_recurrent_kernel_0^ReadVariableOp*
_output_shapes
:	@�*
dtype0f
strided_slice_1/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_1/stack_1Const*
valueB"    �   *
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

:@@*
Index0*
T0m
MatMul_5MatMulplaceholder_2strided_slice_1:output:0*'
_output_shapes
:���������@*
T0h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:���������@L
Const_4Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_5Const*
_output_shapes
: *
valueB
 *   ?*
dtype0[
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:���������@[
Add_3Add	Mul_1:z:0Const_5:output:0*'
_output_shapes
:���������@*
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
:���������@V
clip_by_value_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:���������@b
mul_2Mulclip_by_value_1:z:0placeholder_3*
T0*'
_output_shapes
:���������@�
ReadVariableOp_2ReadVariableOp&readvariableop_lstm_recurrent_kernel_0^ReadVariableOp_1*
_output_shapes
:	@�*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
valueB"    �   *
dtype0h
strided_slice_2/stack_1Const*
valueB"    �   *
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
end_mask*
_output_shapes

:@@*
Index0*
T0m
MatMul_6MatMulplaceholder_2strided_slice_2:output:0*
T0*'
_output_shapes
:���������@h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:���������@I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:���������@[
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:���������@V
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:���������@�
ReadVariableOp_3ReadVariableOp&readvariableop_lstm_recurrent_kernel_0^ReadVariableOp_2*
dtype0*
_output_shapes
:	@�f
strided_slice_3/stackConst*
valueB"    �   *
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
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:@@m
MatMul_7MatMulplaceholder_2strided_slice_3:output:0*
T0*'
_output_shapes
:���������@h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*'
_output_shapes
:���������@*
T0L
Const_6Const*
dtype0*
_output_shapes
: *
valueB
 *��L>L
Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:���������@[
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:���������@^
clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*'
_output_shapes
:���������@*
T0V
clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:���������@K
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:���������@_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:���������@�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_5:z:0*
element_dtype0*
_output_shapes
: I
add_8/yConst*
dtype0*
_output_shapes
: *
value	B :N
add_8AddV2placeholderadd_8/y:output:0*
T0*
_output_shapes
: I
add_9/yConst*
value	B :*
dtype0*
_output_shapes
: e
add_9AddV2"sequential_lstm_while_loop_counteradd_9/y:output:0*
T0*
_output_shapes
: �
IdentityIdentity	add_9:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_1Identity(sequential_lstm_while_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_2Identity	add_8:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_4Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*'
_output_shapes
:���������@*
T0�

Identity_5Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*'
_output_shapes
:���������@*
T0"D
sequential_lstm_strided_slice_1!sequential_lstm_strided_slice_1_0"F
 split_1_readvariableop_lstm_bias"split_1_readvariableop_lstm_bias_0"!

identity_1Identity_1:output:0"�
[tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor]tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"N
$readvariableop_lstm_recurrent_kernel&readvariableop_lstm_recurrent_kernel_0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"F
 split_readvariableop_lstm_kernel"split_readvariableop_lstm_kernel_0*Q
_input_shapes@
>: : : : :���������@:���������@: : :::2 
ReadVariableOpReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp:  : : : : : : : : :	 :
 
�
�
lstm_1_while_cond_6595
lstm_1_while_loop_counter#
lstm_1_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_lstm_1_strided_slice_12
.lstm_1_tensorarrayunstack_tensorlistfromtensor
lstm_1_kernel
lstm_1_bias
lstm_1_recurrent_kernel
identity
W
LessLessplaceholderless_lstm_1_strided_slice_1*
_output_shapes
: *
T0?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :��������� :��������� : : ::::  : : : : : : : : :	 :
 
�
�
while_cond_8378
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
lstm_kernel
	lstm_bias
lstm_recurrent_kernel
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
identityIdentity:output:0*Q
_input_shapes@
>: : : : :���������@:���������@: : :::: : : :	 :
 :  : : : : : 
�
�
?__inference_dense_layer_call_and_return_conditional_losses_9707

inputs&
"matmul_readvariableop_dense_kernel%
!biasadd_readvariableop_dense_bias
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpx
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel*
dtype0*
_output_shapes

: i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:���������*
T0t
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
� 
�
D__inference_sequential_layer_call_and_return_conditional_losses_6089
features_input8
4features_statefulpartitionedcall_features_embeddings,
(lstm_statefulpartitionedcall_lstm_kernel*
&lstm_statefulpartitionedcall_lstm_bias6
2lstm_statefulpartitionedcall_lstm_recurrent_kernel0
,lstm_1_statefulpartitionedcall_lstm_1_kernel.
*lstm_1_statefulpartitionedcall_lstm_1_bias:
6lstm_1_statefulpartitionedcall_lstm_1_recurrent_kernel.
*dense_statefulpartitionedcall_dense_kernel,
(dense_statefulpartitionedcall_dense_bias2
.dense_1_statefulpartitionedcall_dense_1_kernel0
,dense_1_statefulpartitionedcall_dense_1_bias
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall� features/StatefulPartitionedCall�lstm/StatefulPartitionedCall�lstm_1/StatefulPartitionedCall�
 features/StatefulPartitionedCallStatefulPartitionedCallfeatures_input4features_statefulpartitionedcall_features_embeddings*
Tin
2*+
_output_shapes
:���������(*+
_gradient_op_typePartitionedCall-4788*K
fFRD
B__inference_features_layer_call_and_return_conditional_losses_4781*
Tout
2**
config_proto

GPU 

CPU2J 8�
lstm/StatefulPartitionedCallStatefulPartitionedCall)features/StatefulPartitionedCall:output:0(lstm_statefulpartitionedcall_lstm_kernel&lstm_statefulpartitionedcall_lstm_bias2lstm_statefulpartitionedcall_lstm_recurrent_kernel*+
_gradient_op_typePartitionedCall-5367*G
fBR@
>__inference_lstm_layer_call_and_return_conditional_losses_5355*
Tout
2**
config_proto

GPU 

CPU2J 8*+
_output_shapes
:���������(@*
Tin
2�
lstm_1/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0,lstm_1_statefulpartitionedcall_lstm_1_kernel*lstm_1_statefulpartitionedcall_lstm_1_bias6lstm_1_statefulpartitionedcall_lstm_1_recurrent_kernel*+
_gradient_op_typePartitionedCall-5947*I
fDRB
@__inference_lstm_1_layer_call_and_return_conditional_losses_5935*
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
:��������� �
dense/StatefulPartitionedCallStatefulPartitionedCall'lstm_1/StatefulPartitionedCall:output:0*dense_statefulpartitionedcall_dense_kernel(dense_statefulpartitionedcall_dense_bias*+
_gradient_op_typePartitionedCall-5976*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_5969*
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
2�
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:���������*+
_gradient_op_typePartitionedCall-6029*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_6016�
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0.dense_1_statefulpartitionedcall_dense_1_kernel,dense_1_statefulpartitionedcall_dense_1_bias*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:���������*
Tin
2*+
_gradient_op_typePartitionedCall-6052*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_6045�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall!^features/StatefulPartitionedCall^lstm/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*R
_input_shapesA
?:���������(:::::::::::2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2D
 features/StatefulPartitionedCall features/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:. *
(
_user_specified_namefeatures_input: : : : : : : : :	 :
 : 
�
�
"__inference_signature_wrapper_6187
features_input/
+statefulpartitionedcall_features_embeddings'
#statefulpartitionedcall_lstm_kernel%
!statefulpartitionedcall_lstm_bias1
-statefulpartitionedcall_lstm_recurrent_kernel)
%statefulpartitionedcall_lstm_1_kernel'
#statefulpartitionedcall_lstm_1_bias3
/statefulpartitionedcall_lstm_1_recurrent_kernel(
$statefulpartitionedcall_dense_kernel&
"statefulpartitionedcall_dense_bias*
&statefulpartitionedcall_dense_1_kernel(
$statefulpartitionedcall_dense_1_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallfeatures_input+statefulpartitionedcall_features_embeddings#statefulpartitionedcall_lstm_kernel!statefulpartitionedcall_lstm_bias-statefulpartitionedcall_lstm_recurrent_kernel%statefulpartitionedcall_lstm_1_kernel#statefulpartitionedcall_lstm_1_bias/statefulpartitionedcall_lstm_1_recurrent_kernel$statefulpartitionedcall_dense_kernel"statefulpartitionedcall_dense_bias&statefulpartitionedcall_dense_1_kernel$statefulpartitionedcall_dense_1_bias*
Tin
2*'
_output_shapes
:���������*+
_gradient_op_typePartitionedCall-6173*(
f#R!
__inference__wrapped_model_3273*
Tout
2**
config_proto

GPU 

CPU2J 8�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:���������*
T0"
identityIdentity:output:0*R
_input_shapesA
?:���������(:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: :. *
(
_user_specified_namefeatures_input: : : : : : : : :	 :
 
�
�
(__inference_lstm_cell_layer_call_fn_9963

inputs
states_0
states_1'
#statefulpartitionedcall_lstm_kernel%
!statefulpartitionedcall_lstm_bias1
-statefulpartitionedcall_lstm_recurrent_kernel
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1#statefulpartitionedcall_lstm_kernel!statefulpartitionedcall_lstm_bias-statefulpartitionedcall_lstm_recurrent_kernel*+
_gradient_op_typePartitionedCall-3507*L
fGRE
C__inference_lstm_cell_layer_call_and_return_conditional_losses_3407*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin

2*M
_output_shapes;
9:���������@:���������@:���������@�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*'
_output_shapes
:���������@*
T0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:���������:���������@:���������@:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0:($
"
_user_specified_name
states/1: : : 
�
�
while_cond_9526
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
lstm_1_kernel
lstm_1_bias
lstm_1_recurrent_kernel
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
identityIdentity:output:0*Q
_input_shapes@
>: : : : :��������� :��������� : : ::::
 :  : : : : : : : : :	 
�`
�
while_body_7805
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0&
"split_readvariableop_lstm_kernel_0&
"split_1_readvariableop_lstm_bias_0*
&readvariableop_lstm_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor$
 split_readvariableop_lstm_kernel$
 split_1_readvariableop_lstm_bias(
$readvariableop_lstm_recurrent_kernel��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"����   �
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:���������G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: x
split/ReadVariableOpReadVariableOp"split_readvariableop_lstm_kernel_0*
dtype0*
_output_shapes
:	��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
	num_split*<
_output_shapes*
(:@:@:@:@*
T0~
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:0*'
_output_shapes
:���������@*
T0�
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:1*
T0*'
_output_shapes
:���������@�
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:2*
T0*'
_output_shapes
:���������@�
MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:3*'
_output_shapes
:���������@*
T0I
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
_output_shapes
: *
value	B : *
dtype0v
split_1/ReadVariableOpReadVariableOp"split_1_readvariableop_lstm_bias_0*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split*,
_output_shapes
:@:@:@:@h
BiasAddBiasAddMatMul:product:0split_1:output:0*'
_output_shapes
:���������@*
T0l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*'
_output_shapes
:���������@*
T0l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:���������@l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*'
_output_shapes
:���������@*
T0v
ReadVariableOpReadVariableOp&readvariableop_lstm_recurrent_kernel_0*
dtype0*
_output_shapes
:	@�d
strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_1Const*
valueB"    @   *
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

:@@*
T0*
Index0k
MatMul_4MatMulplaceholder_2strided_slice:output:0*
T0*'
_output_shapes
:���������@d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:���������@L
Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *��L>L
Const_3Const*
valueB
 *   ?*
dtype0*
_output_shapes
: W
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:���������@Y
Add_1AddMul:z:0Const_3:output:0*'
_output_shapes
:���������@*
T0\
clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*'
_output_shapes
:���������@*
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
:���������@�
ReadVariableOp_1ReadVariableOp&readvariableop_lstm_recurrent_kernel_0^ReadVariableOp*
_output_shapes
:	@�*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
valueB"    @   *
dtype0h
strided_slice_2/stack_1Const*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
strided_slice_2StridedSliceReadVariableOp_1:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
end_mask*
_output_shapes

:@@*
T0*
Index0*

begin_maskm
MatMul_5MatMulplaceholder_2strided_slice_2:output:0*'
_output_shapes
:���������@*
T0h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:���������@L
Const_4Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:���������@[
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*'
_output_shapes
:���������@^
clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*'
_output_shapes
:���������@*
T0V
clip_by_value_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*'
_output_shapes
:���������@*
T0b
mul_2Mulclip_by_value_1:z:0placeholder_3*
T0*'
_output_shapes
:���������@�
ReadVariableOp_2ReadVariableOp&readvariableop_lstm_recurrent_kernel_0^ReadVariableOp_1*
dtype0*
_output_shapes
:	@�f
strided_slice_3/stackConst*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB"    �   h
strided_slice_3/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_3StridedSliceReadVariableOp_2:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:@@*
T0*
Index0m
MatMul_6MatMulplaceholder_2strided_slice_3:output:0*
T0*'
_output_shapes
:���������@h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:���������@I
TanhTanh	add_4:z:0*'
_output_shapes
:���������@*
T0[
mul_3Mulclip_by_value:z:0Tanh:y:0*'
_output_shapes
:���������@*
T0V
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:���������@�
ReadVariableOp_3ReadVariableOp&readvariableop_lstm_recurrent_kernel_0^ReadVariableOp_2*
dtype0*
_output_shapes
:	@�f
strided_slice_4/stackConst*
dtype0*
_output_shapes
:*
valueB"    �   h
strided_slice_4/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_4StridedSliceReadVariableOp_3:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:@@*
T0*
Index0m
MatMul_7MatMulplaceholder_2strided_slice_4:output:0*'
_output_shapes
:���������@*
T0h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:���������@L
Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_7Const*
_output_shapes
: *
valueB
 *   ?*
dtype0[
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:���������@[
Add_7Add	Mul_4:z:0Const_7:output:0*'
_output_shapes
:���������@*
T0^
clip_by_value_2/Minimum/yConst*
dtype0*
_output_shapes
: *
valueB
 *  �?�
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:���������@V
clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:���������@K
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:���������@_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*'
_output_shapes
:���������@*
T0�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_5:z:0*
element_dtype0*
_output_shapes
: I
add_8/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_8AddV2placeholderadd_8/y:output:0*
T0*
_output_shapes
: I
add_9/yConst*
dtype0*
_output_shapes
: *
value	B :U
add_9AddV2while_loop_counteradd_9/y:output:0*
T0*
_output_shapes
: �
IdentityIdentity	add_9:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_1Identitywhile_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_2Identity	add_8:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_4Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:���������@�

Identity_5Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*'
_output_shapes
:���������@*
T0"F
 split_1_readvariableop_lstm_bias"split_1_readvariableop_lstm_bias_0"!

identity_1Identity_1:output:0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"N
$readvariableop_lstm_recurrent_kernel&readvariableop_lstm_recurrent_kernel_0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 split_readvariableop_lstm_kernel"split_readvariableop_lstm_kernel_0"$
strided_slice_1strided_slice_1_0*Q
_input_shapes@
>: : : : :���������@:���������@: : :::2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp2 
ReadVariableOpReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp: : : : : : : :	 :
 :  : 
�`
�
while_body_5782
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0(
$split_readvariableop_lstm_1_kernel_0(
$split_1_readvariableop_lstm_1_bias_0,
(readvariableop_lstm_1_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor&
"split_readvariableop_lstm_1_kernel&
"split_1_readvariableop_lstm_1_bias*
&readvariableop_lstm_1_recurrent_kernel��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
valueB"����@   *
dtype0�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:���������@G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: z
split/ReadVariableOpReadVariableOp$split_readvariableop_lstm_1_kernel_0*
dtype0*
_output_shapes
:	@��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split*<
_output_shapes*
(:@ :@ :@ :@ ~
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:0*
T0*'
_output_shapes
:��������� �
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:1*
T0*'
_output_shapes
:��������� �
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:2*
T0*'
_output_shapes
:��������� �
MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:3*'
_output_shapes
:��������� *
T0I
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: x
split_1/ReadVariableOpReadVariableOp$split_1_readvariableop_lstm_1_bias_0*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
	num_split*,
_output_shapes
: : : : *
T0h
BiasAddBiasAddMatMul:product:0split_1:output:0*'
_output_shapes
:��������� *
T0l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:��������� l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:��������� l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:��������� x
ReadVariableOpReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0*
dtype0*
_output_shapes
:	 �d
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
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:  k
MatMul_4MatMulplaceholder_2strided_slice:output:0*
T0*'
_output_shapes
:��������� d
addAddV2BiasAdd:output:0MatMul_4:product:0*
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
dtype0W
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:��������� Y
Add_1AddMul:z:0Const_3:output:0*
T0*'
_output_shapes
:��������� \
clip_by_value/Minimum/yConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
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
:��������� �
ReadVariableOp_1ReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0^ReadVariableOp*
dtype0*
_output_shapes
:	 �f
strided_slice_2/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB"    @   h
strided_slice_2/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_2StridedSliceReadVariableOp_1:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
T0*
Index0m
MatMul_5MatMulplaceholder_2strided_slice_2:output:0*'
_output_shapes
:��������� *
T0h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:��������� L
Const_4Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_4:output:0*'
_output_shapes
:��������� *
T0[
Add_3Add	Mul_1:z:0Const_5:output:0*
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
T0b
mul_2Mulclip_by_value_1:z:0placeholder_3*
T0*'
_output_shapes
:��������� �
ReadVariableOp_2ReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0^ReadVariableOp_1*
dtype0*
_output_shapes
:	 �f
strided_slice_3/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_3/stack_1Const*
_output_shapes
:*
valueB"    `   *
dtype0h
strided_slice_3/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_3StridedSliceReadVariableOp_2:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
Index0*
T0m
MatMul_6MatMulplaceholder_2strided_slice_3:output:0*'
_output_shapes
:��������� *
T0h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*'
_output_shapes
:��������� *
T0I
TanhTanh	add_4:z:0*'
_output_shapes
:��������� *
T0[
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_3ReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0^ReadVariableOp_2*
_output_shapes
:	 �*
dtype0f
strided_slice_4/stackConst*
valueB"    `   *
dtype0*
_output_shapes
:h
strided_slice_4/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_4StridedSliceReadVariableOp_3:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
end_mask*
_output_shapes

:  *
Index0*
T0*

begin_maskm
MatMul_7MatMulplaceholder_2strided_slice_4:output:0*'
_output_shapes
:��������� *
T0h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:��������� L
Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:��������� [
Add_7Add	Mul_4:z:0Const_7:output:0*'
_output_shapes
:��������� *
T0^
clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:��������� V
clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*'
_output_shapes
:��������� *
T0K
Tanh_1Tanh	add_5:z:0*'
_output_shapes
:��������� *
T0_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*'
_output_shapes
:��������� *
T0�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_5:z:0*
element_dtype0*
_output_shapes
: I
add_8/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_8AddV2placeholderadd_8/y:output:0*
T0*
_output_shapes
: I
add_9/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_9AddV2while_loop_counteradd_9/y:output:0*
T0*
_output_shapes
: �
IdentityIdentity	add_9:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_1Identitywhile_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_2Identity	add_8:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_4Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:��������� �

Identity_5Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*'
_output_shapes
:��������� *
T0"$
strided_slice_1strided_slice_1_0"J
"split_readvariableop_lstm_1_kernel$split_readvariableop_lstm_1_kernel_0"R
&readvariableop_lstm_1_recurrent_kernel(readvariableop_lstm_1_recurrent_kernel_0"!

identity_1Identity_1:output:0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"J
"split_1_readvariableop_lstm_1_bias$split_1_readvariableop_lstm_1_bias_0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0*Q
_input_shapes@
>: : : : :��������� :��������� : : :::20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp2 
ReadVariableOpReadVariableOp:  : : : : : : : : :	 :
 
��
�
@__inference_lstm_1_layer_call_and_return_conditional_losses_5656

inputs&
"split_readvariableop_lstm_1_kernel&
"split_1_readvariableop_lstm_1_bias*
&readvariableop_lstm_1_recurrent_kernel
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOp�while;
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
_output_shapes
: *
T0*
Index0*
shrink_axis_maskM
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
:��������� O
zeros_1/mul/yConst*
value	B : *
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
dtype0*
_output_shapes
: *
value
B :�_
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B : *
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*'
_output_shapes
:��������� *
T0c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*+
_output_shapes
:(���������@*
T0D
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
_output_shapes
: *
T0*
Index0*
shrink_axis_maskf
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
valueB"����@   *
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
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:���������@G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: x
split/ReadVariableOpReadVariableOp"split_readvariableop_lstm_1_kernel*
dtype0*
_output_shapes
:	@��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split*<
_output_shapes*
(:@ :@ :@ :@ l
MatMulMatMulstrided_slice_2:output:0split:output:0*
T0*'
_output_shapes
:��������� n
MatMul_1MatMulstrided_slice_2:output:0split:output:1*
T0*'
_output_shapes
:��������� n
MatMul_2MatMulstrided_slice_2:output:0split:output:2*
T0*'
_output_shapes
:��������� n
MatMul_3MatMulstrided_slice_2:output:0split:output:3*'
_output_shapes
:��������� *
T0I
Const_1Const*
_output_shapes
: *
value	B :*
dtype0S
split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: v
split_1/ReadVariableOpReadVariableOp"split_1_readvariableop_lstm_1_bias*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split*,
_output_shapes
: : : : h
BiasAddBiasAddMatMul:product:0split_1:output:0*'
_output_shapes
:��������� *
T0l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:��������� l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:��������� l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:��������� v
ReadVariableOpReadVariableOp&readvariableop_lstm_1_recurrent_kernel*
_output_shapes
:	 �*
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

:  n
MatMul_4MatMulzeros:output:0strided_slice_3:output:0*
T0*'
_output_shapes
:��������� d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:��������� L
Const_2Const*
_output_shapes
: *
valueB
 *��L>*
dtype0L
Const_3Const*
_output_shapes
: *
valueB
 *   ?*
dtype0W
MulMuladd:z:0Const_2:output:0*'
_output_shapes
:��������� *
T0Y
Add_1AddMul:z:0Const_3:output:0*'
_output_shapes
:��������� *
T0\
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
:��������� �
ReadVariableOp_1ReadVariableOp&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp*
_output_shapes
:	 �*
dtype0f
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
strided_slice_4StridedSliceReadVariableOp_1:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
_output_shapes

:  *
Index0*
T0*

begin_mask*
end_maskn
MatMul_5MatMulzeros:output:0strided_slice_4:output:0*
T0*'
_output_shapes
:��������� h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*'
_output_shapes
:��������� *
T0L
Const_4Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_4:output:0*'
_output_shapes
:��������� *
T0[
Add_3Add	Mul_1:z:0Const_5:output:0*'
_output_shapes
:��������� *
T0^
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
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:��������� e
mul_2Mulclip_by_value_1:z:0zeros_1:output:0*'
_output_shapes
:��������� *
T0�
ReadVariableOp_2ReadVariableOp&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp_1*
_output_shapes
:	 �*
dtype0f
strided_slice_5/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_5/stack_1Const*
dtype0*
_output_shapes
:*
valueB"    `   h
strided_slice_5/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_5StridedSliceReadVariableOp_2:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:  n
MatMul_6MatMulzeros:output:0strided_slice_5:output:0*'
_output_shapes
:��������� *
T0h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_4:z:0*'
_output_shapes
:��������� *
T0[
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_5AddV2	mul_2:z:0	mul_3:z:0*'
_output_shapes
:��������� *
T0�
ReadVariableOp_3ReadVariableOp&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp_2*
dtype0*
_output_shapes
:	 �f
strided_slice_6/stackConst*
valueB"    `   *
dtype0*
_output_shapes
:h
strided_slice_6/stack_1Const*
dtype0*
_output_shapes
:*
valueB"        h
strided_slice_6/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_slice_6StridedSliceReadVariableOp_3:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
Index0*
T0n
MatMul_7MatMulzeros:output:0strided_slice_6:output:0*'
_output_shapes
:��������� *
T0h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*'
_output_shapes
:��������� *
T0L
Const_6Const*
_output_shapes
: *
valueB
 *��L>*
dtype0L
Const_7Const*
_output_shapes
: *
valueB
 *   ?*
dtype0[
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:��������� [
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:��������� ^
clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:��������� V
clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*'
_output_shapes
:��������� *
T0K
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:��������� _
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*'
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
timeConst*
_output_shapes
: *
value	B : *
dtype0c
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"split_readvariableop_lstm_1_kernel"split_1_readvariableop_lstm_1_bias&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T
2*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_5502*
_num_original_outputs*
bodyR
while_body_5503*L
_output_shapes:
8: : : : :��������� :��������� : : : : : K
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
T0^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:��������� M
while/Identity_6Identitywhile:output:6*
_output_shapes
: *
T0M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
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
:(��������� h
strided_slice_7/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:a
strided_slice_7/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_7StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*'
_output_shapes
:��������� *
T0*
Index0*
shrink_axis_maske
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*+
_output_shapes
:���������( *
T0�
IdentityIdentitystrided_slice_7:output:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp^while*'
_output_shapes
:��������� *
T0"
identityIdentity:output:0*6
_input_shapes%
#:���������(@:::20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp2
whilewhile2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs: : : 
��
�
>__inference_lstm_layer_call_and_return_conditional_losses_5355

inputs$
 split_readvariableop_lstm_kernel$
 split_1_readvariableop_lstm_bias(
$readvariableop_lstm_recurrent_kernel
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOp�while;
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
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
_output_shapes
: *
value	B :@*
dtype0_
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
value	B :@s
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
:���������@O
zeros_1/mul/yConst*
_output_shapes
: *
value	B :@*
dtype0c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
_output_shapes
: *
value	B :@*
dtype0w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
_output_shapes
:*
T0R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*'
_output_shapes
:���������@*
T0c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*+
_output_shapes
:(���������*
T0D
Shape_1Shapetranspose:y:0*
_output_shapes
:*
T0_
strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB: a
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
strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB: a
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
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:���������G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: v
split/ReadVariableOpReadVariableOp split_readvariableop_lstm_kernel*
dtype0*
_output_shapes
:	��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split*<
_output_shapes*
(:@:@:@:@l
MatMulMatMulstrided_slice_2:output:0split:output:0*'
_output_shapes
:���������@*
T0n
MatMul_1MatMulstrided_slice_2:output:0split:output:1*
T0*'
_output_shapes
:���������@n
MatMul_2MatMulstrided_slice_2:output:0split:output:2*
T0*'
_output_shapes
:���������@n
MatMul_3MatMulstrided_slice_2:output:0split:output:3*'
_output_shapes
:���������@*
T0I
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: t
split_1/ReadVariableOpReadVariableOp split_1_readvariableop_lstm_bias*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
	num_split*,
_output_shapes
:@:@:@:@*
T0h
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:���������@l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:���������@l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*'
_output_shapes
:���������@*
T0l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*'
_output_shapes
:���������@*
T0t
ReadVariableOpReadVariableOp$readvariableop_lstm_recurrent_kernel*
dtype0*
_output_shapes
:	@�f
strided_slice_3/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_3/stack_1Const*
valueB"    @   *
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

:@@n
MatMul_4MatMulzeros:output:0strided_slice_3:output:0*'
_output_shapes
:���������@*
T0d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:���������@L
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
: W
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:���������@Y
Add_1AddMul:z:0Const_3:output:0*'
_output_shapes
:���������@*
T0\
clip_by_value/Minimum/yConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*'
_output_shapes
:���������@*
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
:���������@�
ReadVariableOp_1ReadVariableOp$readvariableop_lstm_recurrent_kernel^ReadVariableOp*
dtype0*
_output_shapes
:	@�f
strided_slice_4/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_4/stack_1Const*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_4StridedSliceReadVariableOp_1:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:@@n
MatMul_5MatMulzeros:output:0strided_slice_4:output:0*
T0*'
_output_shapes
:���������@h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*'
_output_shapes
:���������@*
T0L
Const_4Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_4:output:0*'
_output_shapes
:���������@*
T0[
Add_3Add	Mul_1:z:0Const_5:output:0*'
_output_shapes
:���������@*
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
:���������@V
clip_by_value_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:���������@e
mul_2Mulclip_by_value_1:z:0zeros_1:output:0*'
_output_shapes
:���������@*
T0�
ReadVariableOp_2ReadVariableOp$readvariableop_lstm_recurrent_kernel^ReadVariableOp_1*
dtype0*
_output_shapes
:	@�f
strided_slice_5/stackConst*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_5/stack_1Const*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_5/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_slice_5StridedSliceReadVariableOp_2:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:@@*
T0*
Index0n
MatMul_6MatMulzeros:output:0strided_slice_5:output:0*
T0*'
_output_shapes
:���������@h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*'
_output_shapes
:���������@*
T0I
TanhTanh	add_4:z:0*'
_output_shapes
:���������@*
T0[
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:���������@V
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:���������@�
ReadVariableOp_3ReadVariableOp$readvariableop_lstm_recurrent_kernel^ReadVariableOp_2*
dtype0*
_output_shapes
:	@�f
strided_slice_6/stackConst*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_6/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_6/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_6StridedSliceReadVariableOp_3:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:@@*
T0*
Index0n
MatMul_7MatMulzeros:output:0strided_slice_6:output:0*'
_output_shapes
:���������@*
T0h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*'
_output_shapes
:���������@*
T0L
Const_6Const*
_output_shapes
: *
valueB
 *��L>*
dtype0L
Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_4Mul	add_6:z:0Const_6:output:0*'
_output_shapes
:���������@*
T0[
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:���������@^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
valueB
 *  �?*
dtype0�
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*'
_output_shapes
:���������@*
T0V
clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:���������@K
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:���������@_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:���������@n
TensorArrayV2_1/element_shapeConst*
valueB"����@   *
dtype0*
_output_shapes
:�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
element_dtype0*
_output_shapes
: *

shape_type0F
timeConst*
_output_shapes
: *
value	B : *
dtype0c
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 split_readvariableop_lstm_kernel split_1_readvariableop_lstm_bias$readvariableop_lstm_recurrent_kernel^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T
2*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_5201*
_num_original_outputs*
bodyR
while_body_5202*L
_output_shapes:
8: : : : :���������@:���������@: : : : : K
while/IdentityIdentitywhile:output:0*
_output_shapes
: *
T0M
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
:���������@^
while/Identity_5Identitywhile:output:5*'
_output_shapes
:���������@*
T0M
while/Identity_6Identitywhile:output:6*
_output_shapes
: *
T0M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
_output_shapes
: *
T0O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"����@   *
dtype0*
_output_shapes
:�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:(���������@h
strided_slice_7/stackConst*
dtype0*
_output_shapes
:*
valueB:
���������a
strided_slice_7/stack_1Const*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_7StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*'
_output_shapes
:���������@e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������(@�
IdentityIdentitytranspose_1:y:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp^while*
T0*+
_output_shapes
:���������(@"
identityIdentity:output:0*6
_input_shapes%
#:���������(:::20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp2
whilewhile2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs: : : 
�L
�
C__inference_lstm_cell_layer_call_and_return_conditional_losses_9949

inputs
states_0
states_1$
 split_readvariableop_lstm_kernel$
 split_1_readvariableop_lstm_bias(
$readvariableop_lstm_recurrent_kernel
identity

identity_1

identity_2��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOpG
ConstConst*
_output_shapes
: *
value	B :*
dtype0Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: v
split/ReadVariableOpReadVariableOp split_readvariableop_lstm_kernel*
dtype0*
_output_shapes
:	��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split*<
_output_shapes*
(:@:@:@:@Z
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:���������@\
MatMul_1MatMulinputssplit:output:1*'
_output_shapes
:���������@*
T0\
MatMul_2MatMulinputssplit:output:2*'
_output_shapes
:���������@*
T0\
MatMul_3MatMulinputssplit:output:3*'
_output_shapes
:���������@*
T0I
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: t
split_1/ReadVariableOpReadVariableOp split_1_readvariableop_lstm_bias*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split*,
_output_shapes
:@:@:@:@h
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:���������@l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:���������@l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:���������@l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:���������@t
ReadVariableOpReadVariableOp$readvariableop_lstm_recurrent_kernel*
dtype0*
_output_shapes
:	@�d
strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_1Const*
_output_shapes
:*
valueB"    @   *
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

:@@*
Index0*
T0f
MatMul_4MatMulstates_0strided_slice:output:0*
T0*'
_output_shapes
:���������@d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:���������@L
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
: W
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:���������@Y
Add_1AddMul:z:0Const_3:output:0*'
_output_shapes
:���������@*
T0\
clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*'
_output_shapes
:���������@*
T0T
clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*'
_output_shapes
:���������@*
T0�
ReadVariableOp_1ReadVariableOp$readvariableop_lstm_recurrent_kernel^ReadVariableOp*
_output_shapes
:	@�*
dtype0f
strided_slice_1/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_1/stack_1Const*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_1/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:@@*
Index0*
T0h
MatMul_5MatMulstates_0strided_slice_1:output:0*
T0*'
_output_shapes
:���������@h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*'
_output_shapes
:���������@*
T0L
Const_4Const*
dtype0*
_output_shapes
: *
valueB
 *��L>L
Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_4:output:0*'
_output_shapes
:���������@*
T0[
Add_3Add	Mul_1:z:0Const_5:output:0*'
_output_shapes
:���������@*
T0^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
valueB
 *  �?*
dtype0�
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:���������@V
clip_by_value_1/yConst*
_output_shapes
: *
valueB
 *    *
dtype0�
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:���������@]
mul_2Mulclip_by_value_1:z:0states_1*
T0*'
_output_shapes
:���������@�
ReadVariableOp_2ReadVariableOp$readvariableop_lstm_recurrent_kernel^ReadVariableOp_1*
dtype0*
_output_shapes
:	@�f
strided_slice_2/stackConst*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_1Const*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
end_mask*
_output_shapes

:@@*
T0*
Index0*

begin_maskh
MatMul_6MatMulstates_0strided_slice_2:output:0*'
_output_shapes
:���������@*
T0h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:���������@I
TanhTanh	add_4:z:0*'
_output_shapes
:���������@*
T0[
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:���������@V
add_5AddV2	mul_2:z:0	mul_3:z:0*'
_output_shapes
:���������@*
T0�
ReadVariableOp_3ReadVariableOp$readvariableop_lstm_recurrent_kernel^ReadVariableOp_2*
dtype0*
_output_shapes
:	@�f
strided_slice_3/stackConst*
valueB"    �   *
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
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
end_mask*
_output_shapes

:@@*
T0*
Index0*

begin_maskh
MatMul_7MatMulstates_0strided_slice_3:output:0*
T0*'
_output_shapes
:���������@h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*'
_output_shapes
:���������@*
T0L
Const_6Const*
_output_shapes
: *
valueB
 *��L>*
dtype0L
Const_7Const*
dtype0*
_output_shapes
: *
valueB
 *   ?[
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:���������@[
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:���������@^
clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*'
_output_shapes
:���������@*
T0V
clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*'
_output_shapes
:���������@*
T0K
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:���������@_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:���������@�
IdentityIdentity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*'
_output_shapes
:���������@*
T0�

Identity_1Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:���������@�

Identity_2Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*'
_output_shapes
:���������@*
T0"!

identity_2Identity_2:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0*X
_input_shapesG
E:���������:���������@:���������@:::2$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp2 
ReadVariableOpReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:($
"
_user_specified_name
states/0:($
"
_user_specified_name
states/1: : : :& "
 
_user_specified_nameinputs
�
�
while_cond_4684
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
lstm_1_kernel
lstm_1_bias
lstm_1_recurrent_kernel
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
identityIdentity:output:0*Q
_input_shapes@
>: : : : :��������� :��������� : : ::::
 :  : : : : : : : : :	 
�B
�
>__inference_lstm_layer_call_and_return_conditional_losses_3878

inputs'
#statefulpartitionedcall_lstm_kernel%
!statefulpartitionedcall_lstm_bias1
-statefulpartitionedcall_lstm_recurrent_kernel
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
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
_output_shapes
: *
T0*
Index0*
shrink_axis_maskM
zeros/mul/yConst*
value	B :@*
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
value	B :@s
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
:���������@O
zeros_1/mul/yConst*
dtype0*
_output_shapes
: *
value	B :@c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
_output_shapes
: *
T0Q
zeros_1/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@c
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
strided_slice_2/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*'
_output_shapes
:���������*
Index0*
T0*
shrink_axis_mask�
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0#statefulpartitionedcall_lstm_kernel!statefulpartitionedcall_lstm_bias-statefulpartitionedcall_lstm_recurrent_kernel*
Tout
2**
config_proto

GPU 

CPU2J 8*M
_output_shapes;
9:���������@:���������@:���������@*
Tin

2*+
_gradient_op_typePartitionedCall-3507*L
fGRE
C__inference_lstm_cell_layer_call_and_return_conditional_losses_3407n
TensorArrayV2_1/element_shapeConst*
valueB"����@   *
dtype0*
_output_shapes
:�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
element_dtype0*
_output_shapes
: *

shape_type0F
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#statefulpartitionedcall_lstm_kernel!statefulpartitionedcall_lstm_bias-statefulpartitionedcall_lstm_recurrent_kernel^StatefulPartitionedCall*
condR
while_cond_3801*
_num_original_outputs*
bodyR
while_body_3802*L
_output_shapes:
8: : : : :���������@:���������@: : : : : *
T
2*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
_lower_using_switch_merge(*
parallel_iterations K
while/IdentityIdentitywhile:output:0*
_output_shapes
: *
T0M
while/Identity_1Identitywhile:output:1*
_output_shapes
: *
T0M
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
:���������@^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:���������@M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
_output_shapes
: *
T0�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"����@   *
dtype0*
_output_shapes
:�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :������������������@h
strided_slice_3/stackConst*
_output_shapes
:*
valueB:
���������*
dtype0a
strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:���������@*
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
 :������������������@�
IdentityIdentitytranspose_1:y:0^StatefulPartitionedCall^while*4
_output_shapes"
 :������������������@*
T0"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile:& "
 
_user_specified_nameinputs: : : 
ǹ
�
D__inference_sequential_layer_call_and_return_conditional_losses_7353

inputs1
-features_embedding_lookup_features_embeddings)
%lstm_split_readvariableop_lstm_kernel)
%lstm_split_1_readvariableop_lstm_bias-
)lstm_readvariableop_lstm_recurrent_kernel-
)lstm_1_split_readvariableop_lstm_1_kernel-
)lstm_1_split_1_readvariableop_lstm_1_bias1
-lstm_1_readvariableop_lstm_1_recurrent_kernel,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias0
,dense_1_matmul_readvariableop_dense_1_kernel/
+dense_1_biasadd_readvariableop_dense_1_bias
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�features/embedding_lookup�lstm/ReadVariableOp�lstm/ReadVariableOp_1�lstm/ReadVariableOp_2�lstm/ReadVariableOp_3�lstm/split/ReadVariableOp�lstm/split_1/ReadVariableOp�
lstm/while�lstm_1/ReadVariableOp�lstm_1/ReadVariableOp_1�lstm_1/ReadVariableOp_2�lstm_1/ReadVariableOp_3�lstm_1/split/ReadVariableOp�lstm_1/split_1/ReadVariableOp�lstm_1/while^
features/CastCastinputs*

SrcT0*

DstT0*'
_output_shapes
:���������(�
features/embedding_lookupResourceGather-features_embedding_lookup_features_embeddingsfeatures/Cast:y:0*@
_class6
42loc:@features/embedding_lookup/features/embeddings*
Tindices0*
dtype0*+
_output_shapes
:���������(�
"features/embedding_lookup/IdentityIdentity"features/embedding_lookup:output:0*
T0*@
_class6
42loc:@features/embedding_lookup/features/embeddings*+
_output_shapes
:���������(�
$features/embedding_lookup/Identity_1Identity+features/embedding_lookup/Identity:output:0*+
_output_shapes
:���������(*
T0g

lstm/ShapeShape-features/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:b
lstm/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:d
lstm/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:d
lstm/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: R
lstm/zeros/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: n
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: T
lstm/zeros/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: h
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
_output_shapes
: *
T0U
lstm/zeros/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: �
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
T0*
N*
_output_shapes
:U
lstm/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: {

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:���������@T
lstm/zeros_1/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: r
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: V
lstm/zeros_1/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: n
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: W
lstm/zeros_1/packed/1Const*
_output_shapes
: *
value	B :@*
dtype0�
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
_output_shapes
:*
T0*
NW
lstm/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: �
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@h
lstm/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
lstm/transpose	Transpose-features/embedding_lookup/Identity_1:output:0lstm/transpose/perm:output:0*
T0*+
_output_shapes
:(���������N
lstm/Shape_1Shapelstm/transpose:y:0*
_output_shapes
:*
T0d
lstm/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:f
lstm/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:f
lstm/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0k
 lstm/TensorArrayV2/element_shapeConst*
valueB :
���������*
dtype0*
_output_shapes
: �
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: �
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: d
lstm/strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:f
lstm/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:f
lstm/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*'
_output_shapes
:���������*
Index0*
T0*
shrink_axis_maskL

lstm/ConstConst*
_output_shapes
: *
value	B :*
dtype0V
lstm/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: �
lstm/split/ReadVariableOpReadVariableOp%lstm_split_readvariableop_lstm_kernel*
dtype0*
_output_shapes
:	��

lstm/splitSplitlstm/split/split_dim:output:0!lstm/split/ReadVariableOp:value:0*
	num_split*<
_output_shapes*
(:@:@:@:@*
T0{
lstm/MatMulMatMullstm/strided_slice_2:output:0lstm/split:output:0*
T0*'
_output_shapes
:���������@}
lstm/MatMul_1MatMullstm/strided_slice_2:output:0lstm/split:output:1*
T0*'
_output_shapes
:���������@}
lstm/MatMul_2MatMullstm/strided_slice_2:output:0lstm/split:output:2*
T0*'
_output_shapes
:���������@}
lstm/MatMul_3MatMullstm/strided_slice_2:output:0lstm/split:output:3*
T0*'
_output_shapes
:���������@N
lstm/Const_1Const*
value	B :*
dtype0*
_output_shapes
: X
lstm/split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: ~
lstm/split_1/ReadVariableOpReadVariableOp%lstm_split_1_readvariableop_lstm_bias*
_output_shapes	
:�*
dtype0�
lstm/split_1Splitlstm/split_1/split_dim:output:0#lstm/split_1/ReadVariableOp:value:0*
T0*
	num_split*,
_output_shapes
:@:@:@:@w
lstm/BiasAddBiasAddlstm/MatMul:product:0lstm/split_1:output:0*
T0*'
_output_shapes
:���������@{
lstm/BiasAdd_1BiasAddlstm/MatMul_1:product:0lstm/split_1:output:1*
T0*'
_output_shapes
:���������@{
lstm/BiasAdd_2BiasAddlstm/MatMul_2:product:0lstm/split_1:output:2*'
_output_shapes
:���������@*
T0{
lstm/BiasAdd_3BiasAddlstm/MatMul_3:product:0lstm/split_1:output:3*'
_output_shapes
:���������@*
T0~
lstm/ReadVariableOpReadVariableOp)lstm_readvariableop_lstm_recurrent_kernel*
_output_shapes
:	@�*
dtype0k
lstm/strided_slice_3/stackConst*
valueB"        *
dtype0*
_output_shapes
:m
lstm/strided_slice_3/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:m
lstm/strided_slice_3/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
lstm/strided_slice_3StridedSlicelstm/ReadVariableOp:value:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:@@*
T0*
Index0}
lstm/MatMul_4MatMullstm/zeros:output:0lstm/strided_slice_3:output:0*'
_output_shapes
:���������@*
T0s
lstm/addAddV2lstm/BiasAdd:output:0lstm/MatMul_4:product:0*
T0*'
_output_shapes
:���������@Q
lstm/Const_2Const*
_output_shapes
: *
valueB
 *��L>*
dtype0Q
lstm/Const_3Const*
valueB
 *   ?*
dtype0*
_output_shapes
: f
lstm/MulMullstm/add:z:0lstm/Const_2:output:0*
T0*'
_output_shapes
:���������@h

lstm/Add_1Addlstm/Mul:z:0lstm/Const_3:output:0*
T0*'
_output_shapes
:���������@a
lstm/clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
lstm/clip_by_value/MinimumMinimumlstm/Add_1:z:0%lstm/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������@Y
lstm/clip_by_value/yConst*
dtype0*
_output_shapes
: *
valueB
 *    �
lstm/clip_by_valueMaximumlstm/clip_by_value/Minimum:z:0lstm/clip_by_value/y:output:0*'
_output_shapes
:���������@*
T0�
lstm/ReadVariableOp_1ReadVariableOp)lstm_readvariableop_lstm_recurrent_kernel^lstm/ReadVariableOp*
dtype0*
_output_shapes
:	@�k
lstm/strided_slice_4/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:m
lstm/strided_slice_4/stack_1Const*
valueB"    �   *
dtype0*
_output_shapes
:m
lstm/strided_slice_4/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
lstm/strided_slice_4StridedSlicelstm/ReadVariableOp_1:value:0#lstm/strided_slice_4/stack:output:0%lstm/strided_slice_4/stack_1:output:0%lstm/strided_slice_4/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:@@}
lstm/MatMul_5MatMullstm/zeros:output:0lstm/strided_slice_4:output:0*'
_output_shapes
:���������@*
T0w

lstm/add_2AddV2lstm/BiasAdd_1:output:0lstm/MatMul_5:product:0*'
_output_shapes
:���������@*
T0Q
lstm/Const_4Const*
valueB
 *��L>*
dtype0*
_output_shapes
: Q
lstm/Const_5Const*
_output_shapes
: *
valueB
 *   ?*
dtype0j

lstm/Mul_1Mullstm/add_2:z:0lstm/Const_4:output:0*
T0*'
_output_shapes
:���������@j

lstm/Add_3Addlstm/Mul_1:z:0lstm/Const_5:output:0*
T0*'
_output_shapes
:���������@c
lstm/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
valueB
 *  �?*
dtype0�
lstm/clip_by_value_1/MinimumMinimumlstm/Add_3:z:0'lstm/clip_by_value_1/Minimum/y:output:0*'
_output_shapes
:���������@*
T0[
lstm/clip_by_value_1/yConst*
_output_shapes
: *
valueB
 *    *
dtype0�
lstm/clip_by_value_1Maximum lstm/clip_by_value_1/Minimum:z:0lstm/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:���������@t

lstm/mul_2Mullstm/clip_by_value_1:z:0lstm/zeros_1:output:0*'
_output_shapes
:���������@*
T0�
lstm/ReadVariableOp_2ReadVariableOp)lstm_readvariableop_lstm_recurrent_kernel^lstm/ReadVariableOp_1*
dtype0*
_output_shapes
:	@�k
lstm/strided_slice_5/stackConst*
valueB"    �   *
dtype0*
_output_shapes
:m
lstm/strided_slice_5/stack_1Const*
valueB"    �   *
dtype0*
_output_shapes
:m
lstm/strided_slice_5/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
lstm/strided_slice_5StridedSlicelstm/ReadVariableOp_2:value:0#lstm/strided_slice_5/stack:output:0%lstm/strided_slice_5/stack_1:output:0%lstm/strided_slice_5/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:@@*
T0*
Index0}
lstm/MatMul_6MatMullstm/zeros:output:0lstm/strided_slice_5:output:0*
T0*'
_output_shapes
:���������@w

lstm/add_4AddV2lstm/BiasAdd_2:output:0lstm/MatMul_6:product:0*
T0*'
_output_shapes
:���������@S
	lstm/TanhTanhlstm/add_4:z:0*
T0*'
_output_shapes
:���������@j

lstm/mul_3Mullstm/clip_by_value:z:0lstm/Tanh:y:0*'
_output_shapes
:���������@*
T0e

lstm/add_5AddV2lstm/mul_2:z:0lstm/mul_3:z:0*
T0*'
_output_shapes
:���������@�
lstm/ReadVariableOp_3ReadVariableOp)lstm_readvariableop_lstm_recurrent_kernel^lstm/ReadVariableOp_2*
dtype0*
_output_shapes
:	@�k
lstm/strided_slice_6/stackConst*
valueB"    �   *
dtype0*
_output_shapes
:m
lstm/strided_slice_6/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:m
lstm/strided_slice_6/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
lstm/strided_slice_6StridedSlicelstm/ReadVariableOp_3:value:0#lstm/strided_slice_6/stack:output:0%lstm/strided_slice_6/stack_1:output:0%lstm/strided_slice_6/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:@@}
lstm/MatMul_7MatMullstm/zeros:output:0lstm/strided_slice_6:output:0*'
_output_shapes
:���������@*
T0w

lstm/add_6AddV2lstm/BiasAdd_3:output:0lstm/MatMul_7:product:0*'
_output_shapes
:���������@*
T0Q
lstm/Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: Q
lstm/Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: j

lstm/Mul_4Mullstm/add_6:z:0lstm/Const_6:output:0*
T0*'
_output_shapes
:���������@j

lstm/Add_7Addlstm/Mul_4:z:0lstm/Const_7:output:0*'
_output_shapes
:���������@*
T0c
lstm/clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
lstm/clip_by_value_2/MinimumMinimumlstm/Add_7:z:0'lstm/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:���������@[
lstm/clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
lstm/clip_by_value_2Maximum lstm/clip_by_value_2/Minimum:z:0lstm/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:���������@U
lstm/Tanh_1Tanhlstm/add_5:z:0*'
_output_shapes
:���������@*
T0n

lstm/mul_5Mullstm/clip_by_value_2:z:0lstm/Tanh_1:y:0*'
_output_shapes
:���������@*
T0s
"lstm/TensorArrayV2_1/element_shapeConst*
valueB"����@   *
dtype0*
_output_shapes
:�
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: K
	lstm/timeConst*
dtype0*
_output_shapes
: *
value	B : h
lstm/while/maximum_iterationsConst*
valueB :
���������*
dtype0*
_output_shapes
: Y
lstm/while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: �

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0%lstm_split_readvariableop_lstm_kernel%lstm_split_1_readvariableop_lstm_bias)lstm_readvariableop_lstm_recurrent_kernel^lstm/ReadVariableOp_3^lstm/split/ReadVariableOp^lstm/split_1/ReadVariableOp*
T
2*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
_lower_using_switch_merge(*
parallel_iterations * 
condR
lstm_while_cond_6909*
_num_original_outputs* 
bodyR
lstm_while_body_6910*L
_output_shapes:
8: : : : :���������@:���������@: : : : : U
lstm/while/IdentityIdentitylstm/while:output:0*
_output_shapes
: *
T0W
lstm/while/Identity_1Identitylstm/while:output:1*
T0*
_output_shapes
: W
lstm/while/Identity_2Identitylstm/while:output:2*
_output_shapes
: *
T0W
lstm/while/Identity_3Identitylstm/while:output:3*
T0*
_output_shapes
: h
lstm/while/Identity_4Identitylstm/while:output:4*'
_output_shapes
:���������@*
T0h
lstm/while/Identity_5Identitylstm/while:output:5*'
_output_shapes
:���������@*
T0W
lstm/while/Identity_6Identitylstm/while:output:6*
T0*
_output_shapes
: W
lstm/while/Identity_7Identitylstm/while:output:7*
T0*
_output_shapes
: W
lstm/while/Identity_8Identitylstm/while:output:8*
T0*
_output_shapes
: W
lstm/while/Identity_9Identitylstm/while:output:9*
T0*
_output_shapes
: Y
lstm/while/Identity_10Identitylstm/while:output:10*
T0*
_output_shapes
: �
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"����@   �
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while/Identity_3:output:0>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:(���������@m
lstm/strided_slice_7/stackConst*
dtype0*
_output_shapes
:*
valueB:
���������f
lstm/strided_slice_7/stack_1Const*
_output_shapes
:*
valueB: *
dtype0f
lstm/strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
lstm/strided_slice_7StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_7/stack:output:0%lstm/strided_slice_7/stack_1:output:0%lstm/strided_slice_7/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:���������@*
Index0*
T0j
lstm/transpose_1/permConst*
_output_shapes
:*!
valueB"          *
dtype0�
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������(@P
lstm_1/ShapeShapelstm/transpose_1:y:0*
T0*
_output_shapes
:d
lstm_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:f
lstm_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:f
lstm_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
lstm_1/strided_sliceStridedSlicelstm_1/Shape:output:0#lstm_1/strided_slice/stack:output:0%lstm_1/strided_slice/stack_1:output:0%lstm_1/strided_slice/stack_2:output:0*
_output_shapes
: *
T0*
Index0*
shrink_axis_maskT
lstm_1/zeros/mul/yConst*
value	B : *
dtype0*
_output_shapes
: t
lstm_1/zeros/mulMullstm_1/strided_slice:output:0lstm_1/zeros/mul/y:output:0*
T0*
_output_shapes
: V
lstm_1/zeros/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: n
lstm_1/zeros/LessLesslstm_1/zeros/mul:z:0lstm_1/zeros/Less/y:output:0*
T0*
_output_shapes
: W
lstm_1/zeros/packed/1Const*
value	B : *
dtype0*
_output_shapes
: �
lstm_1/zeros/packedPacklstm_1/strided_slice:output:0lstm_1/zeros/packed/1:output:0*
T0*
N*
_output_shapes
:W
lstm_1/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: �
lstm_1/zerosFilllstm_1/zeros/packed:output:0lstm_1/zeros/Const:output:0*'
_output_shapes
:��������� *
T0V
lstm_1/zeros_1/mul/yConst*
value	B : *
dtype0*
_output_shapes
: x
lstm_1/zeros_1/mulMullstm_1/strided_slice:output:0lstm_1/zeros_1/mul/y:output:0*
_output_shapes
: *
T0X
lstm_1/zeros_1/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: t
lstm_1/zeros_1/LessLesslstm_1/zeros_1/mul:z:0lstm_1/zeros_1/Less/y:output:0*
T0*
_output_shapes
: Y
lstm_1/zeros_1/packed/1Const*
value	B : *
dtype0*
_output_shapes
: �
lstm_1/zeros_1/packedPacklstm_1/strided_slice:output:0 lstm_1/zeros_1/packed/1:output:0*
_output_shapes
:*
T0*
NY
lstm_1/zeros_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    �
lstm_1/zeros_1Filllstm_1/zeros_1/packed:output:0lstm_1/zeros_1/Const:output:0*'
_output_shapes
:��������� *
T0j
lstm_1/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
lstm_1/transpose	Transposelstm/transpose_1:y:0lstm_1/transpose/perm:output:0*
T0*+
_output_shapes
:(���������@R
lstm_1/Shape_1Shapelstm_1/transpose:y:0*
T0*
_output_shapes
:f
lstm_1/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:h
lstm_1/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:h
lstm_1/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
lstm_1/strided_slice_1StridedSlicelstm_1/Shape_1:output:0%lstm_1/strided_slice_1/stack:output:0'lstm_1/strided_slice_1/stack_1:output:0'lstm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: m
"lstm_1/TensorArrayV2/element_shapeConst*
valueB :
���������*
dtype0*
_output_shapes
: �
lstm_1/TensorArrayV2TensorListReserve+lstm_1/TensorArrayV2/element_shape:output:0lstm_1/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: �
<lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"����@   *
dtype0*
_output_shapes
:�
.lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_1/transpose:y:0Elstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
element_dtype0*
_output_shapes
: *

shape_type0f
lstm_1/strided_slice_2/stackConst*
_output_shapes
:*
valueB: *
dtype0h
lstm_1/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:h
lstm_1/strided_slice_2/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
lstm_1/strided_slice_2StridedSlicelstm_1/transpose:y:0%lstm_1/strided_slice_2/stack:output:0'lstm_1/strided_slice_2/stack_1:output:0'lstm_1/strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:���������@*
T0*
Index0N
lstm_1/ConstConst*
value	B :*
dtype0*
_output_shapes
: X
lstm_1/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: �
lstm_1/split/ReadVariableOpReadVariableOp)lstm_1_split_readvariableop_lstm_1_kernel*
dtype0*
_output_shapes
:	@��
lstm_1/splitSplitlstm_1/split/split_dim:output:0#lstm_1/split/ReadVariableOp:value:0*
	num_split*<
_output_shapes*
(:@ :@ :@ :@ *
T0�
lstm_1/MatMulMatMullstm_1/strided_slice_2:output:0lstm_1/split:output:0*
T0*'
_output_shapes
:��������� �
lstm_1/MatMul_1MatMullstm_1/strided_slice_2:output:0lstm_1/split:output:1*'
_output_shapes
:��������� *
T0�
lstm_1/MatMul_2MatMullstm_1/strided_slice_2:output:0lstm_1/split:output:2*'
_output_shapes
:��������� *
T0�
lstm_1/MatMul_3MatMullstm_1/strided_slice_2:output:0lstm_1/split:output:3*'
_output_shapes
:��������� *
T0P
lstm_1/Const_1Const*
value	B :*
dtype0*
_output_shapes
: Z
lstm_1/split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: �
lstm_1/split_1/ReadVariableOpReadVariableOp)lstm_1_split_1_readvariableop_lstm_1_bias*
dtype0*
_output_shapes	
:��
lstm_1/split_1Split!lstm_1/split_1/split_dim:output:0%lstm_1/split_1/ReadVariableOp:value:0*
	num_split*,
_output_shapes
: : : : *
T0}
lstm_1/BiasAddBiasAddlstm_1/MatMul:product:0lstm_1/split_1:output:0*
T0*'
_output_shapes
:��������� �
lstm_1/BiasAdd_1BiasAddlstm_1/MatMul_1:product:0lstm_1/split_1:output:1*
T0*'
_output_shapes
:��������� �
lstm_1/BiasAdd_2BiasAddlstm_1/MatMul_2:product:0lstm_1/split_1:output:2*'
_output_shapes
:��������� *
T0�
lstm_1/BiasAdd_3BiasAddlstm_1/MatMul_3:product:0lstm_1/split_1:output:3*'
_output_shapes
:��������� *
T0�
lstm_1/ReadVariableOpReadVariableOp-lstm_1_readvariableop_lstm_1_recurrent_kernel*
dtype0*
_output_shapes
:	 �m
lstm_1/strided_slice_3/stackConst*
valueB"        *
dtype0*
_output_shapes
:o
lstm_1/strided_slice_3/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:o
lstm_1/strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
lstm_1/strided_slice_3StridedSlicelstm_1/ReadVariableOp:value:0%lstm_1/strided_slice_3/stack:output:0'lstm_1/strided_slice_3/stack_1:output:0'lstm_1/strided_slice_3/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
T0*
Index0�
lstm_1/MatMul_4MatMullstm_1/zeros:output:0lstm_1/strided_slice_3:output:0*
T0*'
_output_shapes
:��������� y

lstm_1/addAddV2lstm_1/BiasAdd:output:0lstm_1/MatMul_4:product:0*
T0*'
_output_shapes
:��������� S
lstm_1/Const_2Const*
valueB
 *��L>*
dtype0*
_output_shapes
: S
lstm_1/Const_3Const*
valueB
 *   ?*
dtype0*
_output_shapes
: l

lstm_1/MulMullstm_1/add:z:0lstm_1/Const_2:output:0*
T0*'
_output_shapes
:��������� n
lstm_1/Add_1Addlstm_1/Mul:z:0lstm_1/Const_3:output:0*
T0*'
_output_shapes
:��������� c
lstm_1/clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
lstm_1/clip_by_value/MinimumMinimumlstm_1/Add_1:z:0'lstm_1/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:��������� [
lstm_1/clip_by_value/yConst*
dtype0*
_output_shapes
: *
valueB
 *    �
lstm_1/clip_by_valueMaximum lstm_1/clip_by_value/Minimum:z:0lstm_1/clip_by_value/y:output:0*
T0*'
_output_shapes
:��������� �
lstm_1/ReadVariableOp_1ReadVariableOp-lstm_1_readvariableop_lstm_1_recurrent_kernel^lstm_1/ReadVariableOp*
dtype0*
_output_shapes
:	 �m
lstm_1/strided_slice_4/stackConst*
valueB"        *
dtype0*
_output_shapes
:o
lstm_1/strided_slice_4/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:o
lstm_1/strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
lstm_1/strided_slice_4StridedSlicelstm_1/ReadVariableOp_1:value:0%lstm_1/strided_slice_4/stack:output:0'lstm_1/strided_slice_4/stack_1:output:0'lstm_1/strided_slice_4/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:  �
lstm_1/MatMul_5MatMullstm_1/zeros:output:0lstm_1/strided_slice_4:output:0*'
_output_shapes
:��������� *
T0}
lstm_1/add_2AddV2lstm_1/BiasAdd_1:output:0lstm_1/MatMul_5:product:0*'
_output_shapes
:��������� *
T0S
lstm_1/Const_4Const*
valueB
 *��L>*
dtype0*
_output_shapes
: S
lstm_1/Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: p
lstm_1/Mul_1Mullstm_1/add_2:z:0lstm_1/Const_4:output:0*
T0*'
_output_shapes
:��������� p
lstm_1/Add_3Addlstm_1/Mul_1:z:0lstm_1/Const_5:output:0*'
_output_shapes
:��������� *
T0e
 lstm_1/clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
lstm_1/clip_by_value_1/MinimumMinimumlstm_1/Add_3:z:0)lstm_1/clip_by_value_1/Minimum/y:output:0*'
_output_shapes
:��������� *
T0]
lstm_1/clip_by_value_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
lstm_1/clip_by_value_1Maximum"lstm_1/clip_by_value_1/Minimum:z:0!lstm_1/clip_by_value_1/y:output:0*'
_output_shapes
:��������� *
T0z
lstm_1/mul_2Mullstm_1/clip_by_value_1:z:0lstm_1/zeros_1:output:0*'
_output_shapes
:��������� *
T0�
lstm_1/ReadVariableOp_2ReadVariableOp-lstm_1_readvariableop_lstm_1_recurrent_kernel^lstm_1/ReadVariableOp_1*
dtype0*
_output_shapes
:	 �m
lstm_1/strided_slice_5/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:o
lstm_1/strided_slice_5/stack_1Const*
valueB"    `   *
dtype0*
_output_shapes
:o
lstm_1/strided_slice_5/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
lstm_1/strided_slice_5StridedSlicelstm_1/ReadVariableOp_2:value:0%lstm_1/strided_slice_5/stack:output:0'lstm_1/strided_slice_5/stack_1:output:0'lstm_1/strided_slice_5/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
Index0*
T0�
lstm_1/MatMul_6MatMullstm_1/zeros:output:0lstm_1/strided_slice_5:output:0*'
_output_shapes
:��������� *
T0}
lstm_1/add_4AddV2lstm_1/BiasAdd_2:output:0lstm_1/MatMul_6:product:0*
T0*'
_output_shapes
:��������� W
lstm_1/TanhTanhlstm_1/add_4:z:0*'
_output_shapes
:��������� *
T0p
lstm_1/mul_3Mullstm_1/clip_by_value:z:0lstm_1/Tanh:y:0*
T0*'
_output_shapes
:��������� k
lstm_1/add_5AddV2lstm_1/mul_2:z:0lstm_1/mul_3:z:0*
T0*'
_output_shapes
:��������� �
lstm_1/ReadVariableOp_3ReadVariableOp-lstm_1_readvariableop_lstm_1_recurrent_kernel^lstm_1/ReadVariableOp_2*
_output_shapes
:	 �*
dtype0m
lstm_1/strided_slice_6/stackConst*
_output_shapes
:*
valueB"    `   *
dtype0o
lstm_1/strided_slice_6/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:o
lstm_1/strided_slice_6/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
lstm_1/strided_slice_6StridedSlicelstm_1/ReadVariableOp_3:value:0%lstm_1/strided_slice_6/stack:output:0'lstm_1/strided_slice_6/stack_1:output:0'lstm_1/strided_slice_6/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:  �
lstm_1/MatMul_7MatMullstm_1/zeros:output:0lstm_1/strided_slice_6:output:0*'
_output_shapes
:��������� *
T0}
lstm_1/add_6AddV2lstm_1/BiasAdd_3:output:0lstm_1/MatMul_7:product:0*
T0*'
_output_shapes
:��������� S
lstm_1/Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: S
lstm_1/Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: p
lstm_1/Mul_4Mullstm_1/add_6:z:0lstm_1/Const_6:output:0*'
_output_shapes
:��������� *
T0p
lstm_1/Add_7Addlstm_1/Mul_4:z:0lstm_1/Const_7:output:0*
T0*'
_output_shapes
:��������� e
 lstm_1/clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
lstm_1/clip_by_value_2/MinimumMinimumlstm_1/Add_7:z:0)lstm_1/clip_by_value_2/Minimum/y:output:0*'
_output_shapes
:��������� *
T0]
lstm_1/clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
lstm_1/clip_by_value_2Maximum"lstm_1/clip_by_value_2/Minimum:z:0!lstm_1/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:��������� Y
lstm_1/Tanh_1Tanhlstm_1/add_5:z:0*'
_output_shapes
:��������� *
T0t
lstm_1/mul_5Mullstm_1/clip_by_value_2:z:0lstm_1/Tanh_1:y:0*'
_output_shapes
:��������� *
T0u
$lstm_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
valueB"����    *
dtype0�
lstm_1/TensorArrayV2_1TensorListReserve-lstm_1/TensorArrayV2_1/element_shape:output:0lstm_1/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: M
lstm_1/timeConst*
dtype0*
_output_shapes
: *
value	B : j
lstm_1/while/maximum_iterationsConst*
valueB :
���������*
dtype0*
_output_shapes
: [
lstm_1/while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: �
lstm_1/whileWhile"lstm_1/while/loop_counter:output:0(lstm_1/while/maximum_iterations:output:0lstm_1/time:output:0lstm_1/TensorArrayV2_1:handle:0lstm_1/zeros:output:0lstm_1/zeros_1:output:0lstm_1/strided_slice_1:output:0>lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_1_split_readvariableop_lstm_1_kernel)lstm_1_split_1_readvariableop_lstm_1_bias-lstm_1_readvariableop_lstm_1_recurrent_kernel^lstm_1/ReadVariableOp_3^lstm_1/split/ReadVariableOp^lstm_1/split_1/ReadVariableOp*
T
2*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
_lower_using_switch_merge(*
parallel_iterations *"
condR
lstm_1_while_cond_7184*
_num_original_outputs*"
bodyR
lstm_1_while_body_7185*L
_output_shapes:
8: : : : :��������� :��������� : : : : : Y
lstm_1/while/IdentityIdentitylstm_1/while:output:0*
_output_shapes
: *
T0[
lstm_1/while/Identity_1Identitylstm_1/while:output:1*
_output_shapes
: *
T0[
lstm_1/while/Identity_2Identitylstm_1/while:output:2*
T0*
_output_shapes
: [
lstm_1/while/Identity_3Identitylstm_1/while:output:3*
_output_shapes
: *
T0l
lstm_1/while/Identity_4Identitylstm_1/while:output:4*
T0*'
_output_shapes
:��������� l
lstm_1/while/Identity_5Identitylstm_1/while:output:5*
T0*'
_output_shapes
:��������� [
lstm_1/while/Identity_6Identitylstm_1/while:output:6*
T0*
_output_shapes
: [
lstm_1/while/Identity_7Identitylstm_1/while:output:7*
_output_shapes
: *
T0[
lstm_1/while/Identity_8Identitylstm_1/while:output:8*
T0*
_output_shapes
: [
lstm_1/while/Identity_9Identitylstm_1/while:output:9*
T0*
_output_shapes
: ]
lstm_1/while/Identity_10Identitylstm_1/while:output:10*
T0*
_output_shapes
: �
7lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"����    *
dtype0*
_output_shapes
:�
)lstm_1/TensorArrayV2Stack/TensorListStackTensorListStack lstm_1/while/Identity_3:output:0@lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:(��������� o
lstm_1/strided_slice_7/stackConst*
_output_shapes
:*
valueB:
���������*
dtype0h
lstm_1/strided_slice_7/stack_1Const*
_output_shapes
:*
valueB: *
dtype0h
lstm_1/strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
lstm_1/strided_slice_7StridedSlice2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_1/strided_slice_7/stack:output:0'lstm_1/strided_slice_7/stack_1:output:0'lstm_1/strided_slice_7/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*'
_output_shapes
:��������� l
lstm_1/transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
lstm_1/transpose_1	Transpose2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������( �
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel*
dtype0*
_output_shapes

: �
dense/MatMulMatMullstm_1/strided_slice_7:output:0#dense/MatMul/ReadVariableOp:value:0*
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

dense/ReluReludense/BiasAdd:output:0*'
_output_shapes
:���������*
T0h
dropout/IdentityIdentitydense/Relu:activations:0*
T0*'
_output_shapes
:����������
dense_1/MatMul/ReadVariableOpReadVariableOp,dense_1_matmul_readvariableop_dense_1_kernel*
dtype0*
_output_shapes

:�
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*'
_output_shapes
:���������*
T0�
dense_1/BiasAdd/ReadVariableOpReadVariableOp+dense_1_biasadd_readvariableop_dense_1_bias*
dtype0*
_output_shapes
:�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentitydense_1/Sigmoid:y:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^features/embedding_lookup^lstm/ReadVariableOp^lstm/ReadVariableOp_1^lstm/ReadVariableOp_2^lstm/ReadVariableOp_3^lstm/split/ReadVariableOp^lstm/split_1/ReadVariableOp^lstm/while^lstm_1/ReadVariableOp^lstm_1/ReadVariableOp_1^lstm_1/ReadVariableOp_2^lstm_1/ReadVariableOp_3^lstm_1/split/ReadVariableOp^lstm_1/split_1/ReadVariableOp^lstm_1/while*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*R
_input_shapesA
?:���������(:::::::::::2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2:
lstm_1/split/ReadVariableOplstm_1/split/ReadVariableOp2.
lstm_1/ReadVariableOplstm_1/ReadVariableOp2>
lstm_1/split_1/ReadVariableOplstm_1/split_1/ReadVariableOp26
lstm/split/ReadVariableOplstm/split/ReadVariableOp2

lstm/while
lstm/while2*
lstm/ReadVariableOplstm/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2:
lstm/split_1/ReadVariableOplstm/split_1/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp22
lstm_1/ReadVariableOp_1lstm_1/ReadVariableOp_122
lstm_1/ReadVariableOp_2lstm_1/ReadVariableOp_222
lstm_1/ReadVariableOp_3lstm_1/ReadVariableOp_32
lstm_1/whilelstm_1/while2.
lstm/ReadVariableOp_1lstm/ReadVariableOp_12.
lstm/ReadVariableOp_2lstm/ReadVariableOp_22.
lstm/ReadVariableOp_3lstm/ReadVariableOp_326
features/embedding_lookupfeatures/embedding_lookup:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : 
�`
�
while_body_9527
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0(
$split_readvariableop_lstm_1_kernel_0(
$split_1_readvariableop_lstm_1_bias_0,
(readvariableop_lstm_1_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor&
"split_readvariableop_lstm_1_kernel&
"split_1_readvariableop_lstm_1_bias*
&readvariableop_lstm_1_recurrent_kernel��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����@   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:���������@G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: z
split/ReadVariableOpReadVariableOp$split_readvariableop_lstm_1_kernel_0*
dtype0*
_output_shapes
:	@��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split*<
_output_shapes*
(:@ :@ :@ :@ ~
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:0*
T0*'
_output_shapes
:��������� �
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:1*
T0*'
_output_shapes
:��������� �
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:2*'
_output_shapes
:��������� *
T0�
MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:3*
T0*'
_output_shapes
:��������� I
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: x
split_1/ReadVariableOpReadVariableOp$split_1_readvariableop_lstm_1_bias_0*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split*,
_output_shapes
: : : : h
BiasAddBiasAddMatMul:product:0split_1:output:0*'
_output_shapes
:��������� *
T0l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:��������� l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*'
_output_shapes
:��������� *
T0l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:��������� x
ReadVariableOpReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0*
dtype0*
_output_shapes
:	 �d
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

:  k
MatMul_4MatMulplaceholder_2strided_slice:output:0*'
_output_shapes
:��������� *
T0d
addAddV2BiasAdd:output:0MatMul_4:product:0*
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
: W
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:��������� Y
Add_1AddMul:z:0Const_3:output:0*'
_output_shapes
:��������� *
T0\
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
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*'
_output_shapes
:��������� *
T0�
ReadVariableOp_1ReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0^ReadVariableOp*
dtype0*
_output_shapes
:	 �f
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

:  *
Index0*
T0*

begin_maskm
MatMul_5MatMulplaceholder_2strided_slice_2:output:0*
T0*'
_output_shapes
:��������� h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:��������� L
Const_4Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:��������� [
Add_3Add	Mul_1:z:0Const_5:output:0*'
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
mul_2Mulclip_by_value_1:z:0placeholder_3*'
_output_shapes
:��������� *
T0�
ReadVariableOp_2ReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0^ReadVariableOp_1*
dtype0*
_output_shapes
:	 �f
strided_slice_3/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_3/stack_1Const*
valueB"    `   *
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
end_mask*
_output_shapes

:  m
MatMul_6MatMulplaceholder_2strided_slice_3:output:0*'
_output_shapes
:��������� *
T0h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*'
_output_shapes
:��������� *
T0I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:��������� [
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_3ReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0^ReadVariableOp_2*
dtype0*
_output_shapes
:	 �f
strided_slice_4/stackConst*
valueB"    `   *
dtype0*
_output_shapes
:h
strided_slice_4/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_4StridedSliceReadVariableOp_3:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
T0*
Index0m
MatMul_7MatMulplaceholder_2strided_slice_4:output:0*'
_output_shapes
:��������� *
T0h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:��������� L
Const_6Const*
dtype0*
_output_shapes
: *
valueB
 *��L>L
Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_4Mul	add_6:z:0Const_6:output:0*'
_output_shapes
:��������� *
T0[
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:��������� ^
clip_by_value_2/Minimum/yConst*
dtype0*
_output_shapes
: *
valueB
 *  �?�
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:��������� V
clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:��������� K
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:��������� _
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:��������� �
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_5:z:0*
element_dtype0*
_output_shapes
: I
add_8/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_8AddV2placeholderadd_8/y:output:0*
_output_shapes
: *
T0I
add_9/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_9AddV2while_loop_counteradd_9/y:output:0*
T0*
_output_shapes
: �
IdentityIdentity	add_9:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_1Identitywhile_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_2Identity	add_8:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_4Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*'
_output_shapes
:��������� *
T0�

Identity_5Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:��������� "$
strided_slice_1strided_slice_1_0"J
"split_readvariableop_lstm_1_kernel$split_readvariableop_lstm_1_kernel_0"R
&readvariableop_lstm_1_recurrent_kernel(readvariableop_lstm_1_recurrent_kernel_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"J
"split_1_readvariableop_lstm_1_bias$split_1_readvariableop_lstm_1_bias_0"
identityIdentity:output:0"!

identity_5Identity_5:output:0*Q
_input_shapes@
>: : : : :��������� :��������� : : :::20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp2 
ReadVariableOpReadVariableOp:
 :  : : : : : : : : :	 
�B
�
@__inference_lstm_1_layer_call_and_return_conditional_losses_4624

inputs)
%statefulpartitionedcall_lstm_1_kernel'
#statefulpartitionedcall_lstm_1_bias3
/statefulpartitionedcall_lstm_1_recurrent_kernel
identity��StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: _
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
zeros/packed/1Const*
_output_shapes
: *
value	B : *
dtype0s
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
:��������� O
zeros_1/mul/yConst*
value	B : *
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B : *
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
_output_shapes
:*
T0R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
dtype0*
_output_shapes
:*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*4
_output_shapes"
 :������������������@*
T0D
Shape_1Shapetranspose:y:0*
_output_shapes
:*
T0_
strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB: a
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
valueB"����@   *
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
strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:���������@*
Index0*
T0�
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0%statefulpartitionedcall_lstm_1_kernel#statefulpartitionedcall_lstm_1_bias/statefulpartitionedcall_lstm_1_recurrent_kernel*+
_gradient_op_typePartitionedCall-4253*N
fIRG
E__inference_lstm_cell_1_layer_call_and_return_conditional_losses_4153*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin

2*M
_output_shapes;
9:��������� :��������� :��������� n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0%statefulpartitionedcall_lstm_1_kernel#statefulpartitionedcall_lstm_1_bias/statefulpartitionedcall_lstm_1_recurrent_kernel^StatefulPartitionedCall*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
T
2*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_4547*
_num_original_outputs*
bodyR
while_body_4548*L
_output_shapes:
8: : : : :��������� :��������� : : : : : K
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
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:��������� ^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:��������� M
while/Identity_6Identitywhile:output:6*
_output_shapes
: *
T0M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
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
strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*'
_output_shapes
:��������� e
transpose_1/permConst*
_output_shapes
:*!
valueB"          *
dtype0�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*4
_output_shapes"
 :������������������ *
T0�
IdentityIdentitystrided_slice_3:output:0^StatefulPartitionedCall^while*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*?
_input_shapes.
,:������������������@:::22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile:& "
 
_user_specified_nameinputs: : : 
�
�
&__inference_dense_1_layer_call_fn_9767

inputs*
&statefulpartitionedcall_dense_1_kernel(
$statefulpartitionedcall_dense_1_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs&statefulpartitionedcall_dense_1_kernel$statefulpartitionedcall_dense_1_bias*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_6045*
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
:���������*+
_gradient_op_typePartitionedCall-6052�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�
�
while_body_4685
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
'statefulpartitionedcall_lstm_1_kernel_0)
%statefulpartitionedcall_lstm_1_bias_05
1statefulpartitionedcall_lstm_1_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
%statefulpartitionedcall_lstm_1_kernel'
#statefulpartitionedcall_lstm_1_bias3
/statefulpartitionedcall_lstm_1_recurrent_kernel��StatefulPartitionedCall�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����@   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:���������@�
StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3'statefulpartitionedcall_lstm_1_kernel_0%statefulpartitionedcall_lstm_1_bias_01statefulpartitionedcall_lstm_1_recurrent_kernel_0**
config_proto

GPU 

CPU2J 8*M
_output_shapes;
9:��������� :��������� :��������� *
Tin

2*+
_gradient_op_typePartitionedCall-4271*N
fIRG
E__inference_lstm_cell_1_layer_call_and_return_conditional_losses_4247*
Tout
2�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder StatefulPartitionedCall:output:0*
element_dtype0*
_output_shapes
: G
add/yConst*
dtype0*
_output_shapes
: *
value	B :J
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

Identity_1Identitywhile_maximum_iterations^StatefulPartitionedCall*
_output_shapes
: *
T0Z

Identity_2Identityadd:z:0^StatefulPartitionedCall*
T0*
_output_shapes
: �

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^StatefulPartitionedCall*
_output_shapes
: *
T0�

Identity_4Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� �

Identity_5Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*'
_output_shapes
:��������� *
T0"P
%statefulpartitionedcall_lstm_1_kernel'statefulpartitionedcall_lstm_1_kernel_0"d
/statefulpartitionedcall_lstm_1_recurrent_kernel1statefulpartitionedcall_lstm_1_recurrent_kernel_0"L
#statefulpartitionedcall_lstm_1_bias%statefulpartitionedcall_lstm_1_bias_0"$
strided_slice_1strided_slice_1_0"!

identity_1Identity_1:output:0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0*Q
_input_shapes@
>: : : : :��������� :��������� : : :::22
StatefulPartitionedCallStatefulPartitionedCall:	 :
 :  : : : : : : : : 
�L
�
E__inference_lstm_cell_1_layer_call_and_return_conditional_losses_4247

inputs

states
states_1&
"split_readvariableop_lstm_1_kernel&
"split_1_readvariableop_lstm_1_bias*
&readvariableop_lstm_1_recurrent_kernel
identity

identity_1

identity_2��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOpG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: x
split/ReadVariableOpReadVariableOp"split_readvariableop_lstm_1_kernel*
_output_shapes
:	@�*
dtype0�
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split*<
_output_shapes*
(:@ :@ :@ :@ Z
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:��������� \
MatMul_1MatMulinputssplit:output:1*'
_output_shapes
:��������� *
T0\
MatMul_2MatMulinputssplit:output:2*'
_output_shapes
:��������� *
T0\
MatMul_3MatMulinputssplit:output:3*'
_output_shapes
:��������� *
T0I
Const_1Const*
dtype0*
_output_shapes
: *
value	B :S
split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: v
split_1/ReadVariableOpReadVariableOp"split_1_readvariableop_lstm_1_bias*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
	num_split*,
_output_shapes
: : : : *
T0h
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:��������� l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*'
_output_shapes
:��������� *
T0l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:��������� l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*'
_output_shapes
:��������� *
T0v
ReadVariableOpReadVariableOp&readvariableop_lstm_1_recurrent_kernel*
dtype0*
_output_shapes
:	 �d
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"        f
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB"        f
strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
end_mask*
_output_shapes

:  *
Index0*
T0*

begin_maskd
MatMul_4MatMulstatesstrided_slice:output:0*
T0*'
_output_shapes
:��������� d
addAddV2BiasAdd:output:0MatMul_4:product:0*
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
: W
MulMuladd:z:0Const_2:output:0*'
_output_shapes
:��������� *
T0Y
Add_1AddMul:z:0Const_3:output:0*'
_output_shapes
:��������� *
T0\
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
:��������� �
ReadVariableOp_1ReadVariableOp&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp*
dtype0*
_output_shapes
:	 �f
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
dtype0*
_output_shapes
:*
valueB"      �
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
_output_shapes

:  *
T0*
Index0*

begin_mask*
end_maskf
MatMul_5MatMulstatesstrided_slice_1:output:0*
T0*'
_output_shapes
:��������� h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:��������� L
Const_4Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_5Const*
dtype0*
_output_shapes
: *
valueB
 *   ?[
Mul_1Mul	add_2:z:0Const_4:output:0*'
_output_shapes
:��������� *
T0[
Add_3Add	Mul_1:z:0Const_5:output:0*
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
:��������� ]
mul_2Mulclip_by_value_1:z:0states_1*
T0*'
_output_shapes
:��������� �
ReadVariableOp_2ReadVariableOp&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp_1*
dtype0*
_output_shapes
:	 �f
strided_slice_2/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_1Const*
_output_shapes
:*
valueB"    `   *
dtype0h
strided_slice_2/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
end_mask*
_output_shapes

:  *
T0*
Index0*

begin_maskf
MatMul_6MatMulstatesstrided_slice_2:output:0*'
_output_shapes
:��������� *
T0h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_4:z:0*'
_output_shapes
:��������� *
T0[
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_5AddV2	mul_2:z:0	mul_3:z:0*'
_output_shapes
:��������� *
T0�
ReadVariableOp_3ReadVariableOp&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp_2*
dtype0*
_output_shapes
:	 �f
strided_slice_3/stackConst*
dtype0*
_output_shapes
:*
valueB"    `   h
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
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
T0*
Index0f
MatMul_7MatMulstatesstrided_slice_3:output:0*'
_output_shapes
:��������� *
T0h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:��������� L
Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_7Const*
_output_shapes
: *
valueB
 *   ?*
dtype0[
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:��������� [
Add_7Add	Mul_4:z:0Const_7:output:0*'
_output_shapes
:��������� *
T0^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
valueB
 *  �?*
dtype0�
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:��������� V
clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:��������� K
Tanh_1Tanh	add_5:z:0*'
_output_shapes
:��������� *
T0_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:��������� �
IdentityIdentity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:��������� �

Identity_1Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*'
_output_shapes
:��������� *
T0�

Identity_2Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:���������@:��������� :��������� :::20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namestates:&"
 
_user_specified_namestates: : : 
�	
�
A__inference_dense_1_layer_call_and_return_conditional_losses_6045

inputs(
$matmul_readvariableop_dense_1_kernel'
#biasadd_readvariableop_dense_1_bias
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpz
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_1_kernel*
dtype0*
_output_shapes

:i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_1_bias*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*'
_output_shapes
:���������*
T0�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:���������*
T0"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
�
)__inference_sequential_layer_call_fn_6169
features_input/
+statefulpartitionedcall_features_embeddings'
#statefulpartitionedcall_lstm_kernel%
!statefulpartitionedcall_lstm_bias1
-statefulpartitionedcall_lstm_recurrent_kernel)
%statefulpartitionedcall_lstm_1_kernel'
#statefulpartitionedcall_lstm_1_bias3
/statefulpartitionedcall_lstm_1_recurrent_kernel(
$statefulpartitionedcall_dense_kernel&
"statefulpartitionedcall_dense_bias*
&statefulpartitionedcall_dense_1_kernel(
$statefulpartitionedcall_dense_1_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallfeatures_input+statefulpartitionedcall_features_embeddings#statefulpartitionedcall_lstm_kernel!statefulpartitionedcall_lstm_bias-statefulpartitionedcall_lstm_recurrent_kernel%statefulpartitionedcall_lstm_1_kernel#statefulpartitionedcall_lstm_1_bias/statefulpartitionedcall_lstm_1_recurrent_kernel$statefulpartitionedcall_dense_kernel"statefulpartitionedcall_dense_bias&statefulpartitionedcall_dense_1_kernel$statefulpartitionedcall_dense_1_bias*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_6154*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:���������*+
_gradient_op_typePartitionedCall-6155�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*R
_input_shapesA
?:���������(:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:. *
(
_user_specified_namefeatures_input: : : : : : : : :	 :
 : 
�
_
A__inference_dropout_layer_call_and_return_conditional_losses_6016

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������"!

identity_1Identity_1:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�L
�
C__inference_lstm_cell_layer_call_and_return_conditional_losses_3501

inputs

states
states_1$
 split_readvariableop_lstm_kernel$
 split_1_readvariableop_lstm_bias(
$readvariableop_lstm_recurrent_kernel
identity

identity_1

identity_2��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOpG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
dtype0*
_output_shapes
: *
value	B :v
split/ReadVariableOpReadVariableOp split_readvariableop_lstm_kernel*
dtype0*
_output_shapes
:	��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split*<
_output_shapes*
(:@:@:@:@Z
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:���������@\
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:���������@\
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:���������@\
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:���������@I
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: t
split_1/ReadVariableOpReadVariableOp split_1_readvariableop_lstm_bias*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split*,
_output_shapes
:@:@:@:@h
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:���������@l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:���������@l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:���������@l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*'
_output_shapes
:���������@*
T0t
ReadVariableOpReadVariableOp$readvariableop_lstm_recurrent_kernel*
dtype0*
_output_shapes
:	@�d
strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_1Const*
valueB"    @   *
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

:@@*
Index0*
T0d
MatMul_4MatMulstatesstrided_slice:output:0*
T0*'
_output_shapes
:���������@d
addAddV2BiasAdd:output:0MatMul_4:product:0*'
_output_shapes
:���������@*
T0L
Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *��L>L
Const_3Const*
dtype0*
_output_shapes
: *
valueB
 *   ?W
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:���������@Y
Add_1AddMul:z:0Const_3:output:0*
T0*'
_output_shapes
:���������@\
clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������@T
clip_by_value/yConst*
_output_shapes
: *
valueB
 *    *
dtype0
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*'
_output_shapes
:���������@*
T0�
ReadVariableOp_1ReadVariableOp$readvariableop_lstm_recurrent_kernel^ReadVariableOp*
dtype0*
_output_shapes
:	@�f
strided_slice_1/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_1/stack_1Const*
valueB"    �   *
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

:@@*
T0*
Index0f
MatMul_5MatMulstatesstrided_slice_1:output:0*
T0*'
_output_shapes
:���������@h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:���������@L
Const_4Const*
_output_shapes
: *
valueB
 *��L>*
dtype0L
Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_4:output:0*'
_output_shapes
:���������@*
T0[
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*'
_output_shapes
:���������@^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
valueB
 *  �?*
dtype0�
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*'
_output_shapes
:���������@*
T0V
clip_by_value_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *    �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:���������@]
mul_2Mulclip_by_value_1:z:0states_1*'
_output_shapes
:���������@*
T0�
ReadVariableOp_2ReadVariableOp$readvariableop_lstm_recurrent_kernel^ReadVariableOp_1*
dtype0*
_output_shapes
:	@�f
strided_slice_2/stackConst*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_1Const*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:@@*
Index0*
T0f
MatMul_6MatMulstatesstrided_slice_2:output:0*'
_output_shapes
:���������@*
T0h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:���������@I
TanhTanh	add_4:z:0*'
_output_shapes
:���������@*
T0[
mul_3Mulclip_by_value:z:0Tanh:y:0*'
_output_shapes
:���������@*
T0V
add_5AddV2	mul_2:z:0	mul_3:z:0*'
_output_shapes
:���������@*
T0�
ReadVariableOp_3ReadVariableOp$readvariableop_lstm_recurrent_kernel^ReadVariableOp_2*
dtype0*
_output_shapes
:	@�f
strided_slice_3/stackConst*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_3/stack_1Const*
_output_shapes
:*
valueB"        *
dtype0h
strided_slice_3/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:@@*
Index0*
T0f
MatMul_7MatMulstatesstrided_slice_3:output:0*'
_output_shapes
:���������@*
T0h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:���������@L
Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:���������@[
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:���������@^
clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*'
_output_shapes
:���������@*
T0V
clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:���������@K
Tanh_1Tanh	add_5:z:0*'
_output_shapes
:���������@*
T0_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:���������@�
IdentityIdentity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*'
_output_shapes
:���������@*
T0�

Identity_1Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*'
_output_shapes
:���������@*
T0�

Identity_2Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:���������:���������@:���������@:::2 
ReadVariableOpReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namestates:&"
 
_user_specified_namestates: : : 
�
�
#__inference_lstm_layer_call_fn_8548

inputs'
#statefulpartitionedcall_lstm_kernel%
!statefulpartitionedcall_lstm_bias1
-statefulpartitionedcall_lstm_recurrent_kernel
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs#statefulpartitionedcall_lstm_kernel!statefulpartitionedcall_lstm_bias-statefulpartitionedcall_lstm_recurrent_kernel*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:���������(@*+
_gradient_op_typePartitionedCall-5367*G
fBR@
>__inference_lstm_layer_call_and_return_conditional_losses_5355�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������(@"
identityIdentity:output:0*6
_input_shapes%
#:���������(:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
�
�
while_cond_7804
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
lstm_kernel
	lstm_bias
lstm_recurrent_kernel
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
identityIdentity:output:0*Q
_input_shapes@
>: : : : :���������@:���������@: : :::: : : : : : :	 :
 :  : : 
�
�
+__inference_lstm_cell_1_layer_call_fn_10173

inputs
states_0
states_1)
%statefulpartitionedcall_lstm_1_kernel'
#statefulpartitionedcall_lstm_1_bias3
/statefulpartitionedcall_lstm_1_recurrent_kernel
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1%statefulpartitionedcall_lstm_1_kernel#statefulpartitionedcall_lstm_1_bias/statefulpartitionedcall_lstm_1_recurrent_kernel**
config_proto

GPU 

CPU2J 8*M
_output_shapes;
9:��������� :��������� :��������� *
Tin

2*+
_gradient_op_typePartitionedCall-4253*N
fIRG
E__inference_lstm_cell_1_layer_call_and_return_conditional_losses_4153*
Tout
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� �

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*'
_output_shapes
:��������� *
T0�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*'
_output_shapes
:��������� *
T0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*X
_input_shapesG
E:���������@:��������� :��������� :::22
StatefulPartitionedCallStatefulPartitionedCall: : :& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0:($
"
_user_specified_name
states/1: 
�
�
%__inference_lstm_1_layer_call_fn_9688

inputs)
%statefulpartitionedcall_lstm_1_kernel'
#statefulpartitionedcall_lstm_1_bias3
/statefulpartitionedcall_lstm_1_recurrent_kernel
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs%statefulpartitionedcall_lstm_1_kernel#statefulpartitionedcall_lstm_1_bias/statefulpartitionedcall_lstm_1_recurrent_kernel**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:��������� *
Tin
2*+
_gradient_op_typePartitionedCall-5938*I
fDRB
@__inference_lstm_1_layer_call_and_return_conditional_losses_5656*
Tout
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:��������� *
T0"
identityIdentity:output:0*6
_input_shapes%
#:���������(@:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
�L
�
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_10068

inputs
states_0
states_1&
"split_readvariableop_lstm_1_kernel&
"split_1_readvariableop_lstm_1_bias*
&readvariableop_lstm_1_recurrent_kernel
identity

identity_1

identity_2��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOpG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
_output_shapes
: *
value	B :*
dtype0x
split/ReadVariableOpReadVariableOp"split_readvariableop_lstm_1_kernel*
dtype0*
_output_shapes
:	@��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
	num_split*<
_output_shapes*
(:@ :@ :@ :@ *
T0Z
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:��������� \
MatMul_1MatMulinputssplit:output:1*'
_output_shapes
:��������� *
T0\
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:��������� \
MatMul_3MatMulinputssplit:output:3*'
_output_shapes
:��������� *
T0I
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: v
split_1/ReadVariableOpReadVariableOp"split_1_readvariableop_lstm_1_bias*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split*,
_output_shapes
: : : : h
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:��������� l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*'
_output_shapes
:��������� *
T0l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:��������� l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*'
_output_shapes
:��������� *
T0v
ReadVariableOpReadVariableOp&readvariableop_lstm_1_recurrent_kernel*
dtype0*
_output_shapes
:	 �d
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
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
T0*
Index0f
MatMul_4MatMulstates_0strided_slice:output:0*
T0*'
_output_shapes
:��������� d
addAddV2BiasAdd:output:0MatMul_4:product:0*
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
 *   ?W
MulMuladd:z:0Const_2:output:0*'
_output_shapes
:��������� *
T0Y
Add_1AddMul:z:0Const_3:output:0*'
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
clip_by_value/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_1ReadVariableOp&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp*
dtype0*
_output_shapes
:	 �f
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

:  *
Index0*
T0h
MatMul_5MatMulstates_0strided_slice_1:output:0*
T0*'
_output_shapes
:��������� h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:��������� L
Const_4Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:��������� [
Add_3Add	Mul_1:z:0Const_5:output:0*'
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
T0]
mul_2Mulclip_by_value_1:z:0states_1*'
_output_shapes
:��������� *
T0�
ReadVariableOp_2ReadVariableOp&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp_1*
_output_shapes
:	 �*
dtype0f
strided_slice_2/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_1Const*
valueB"    `   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
end_mask*
_output_shapes

:  *
T0*
Index0*

begin_maskh
MatMul_6MatMulstates_0strided_slice_2:output:0*'
_output_shapes
:��������� *
T0h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*'
_output_shapes
:��������� *
T0I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:��������� [
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_3ReadVariableOp&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp_2*
dtype0*
_output_shapes
:	 �f
strided_slice_3/stackConst*
_output_shapes
:*
valueB"    `   *
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
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
Index0*
T0h
MatMul_7MatMulstates_0strided_slice_3:output:0*
T0*'
_output_shapes
:��������� h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:��������� L
Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:��������� [
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:��������� ^
clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:��������� V
clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*'
_output_shapes
:��������� *
T0K
Tanh_1Tanh	add_5:z:0*'
_output_shapes
:��������� *
T0_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*'
_output_shapes
:��������� *
T0�
IdentityIdentity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*'
_output_shapes
:��������� *
T0�

Identity_1Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:��������� �

Identity_2Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*X
_input_shapesG
E:���������@:��������� :��������� :::2 
ReadVariableOpReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp: :& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0:($
"
_user_specified_name
states/1: : 
�
�
while_body_4548
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
'statefulpartitionedcall_lstm_1_kernel_0)
%statefulpartitionedcall_lstm_1_bias_05
1statefulpartitionedcall_lstm_1_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
%statefulpartitionedcall_lstm_1_kernel'
#statefulpartitionedcall_lstm_1_bias3
/statefulpartitionedcall_lstm_1_recurrent_kernel��StatefulPartitionedCall�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����@   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:���������@�
StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3'statefulpartitionedcall_lstm_1_kernel_0%statefulpartitionedcall_lstm_1_bias_01statefulpartitionedcall_lstm_1_recurrent_kernel_0*
Tin

2*M
_output_shapes;
9:��������� :��������� :��������� *+
_gradient_op_typePartitionedCall-4253*N
fIRG
E__inference_lstm_cell_1_layer_call_and_return_conditional_losses_4153*
Tout
2**
config_proto

GPU 

CPU2J 8�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder StatefulPartitionedCall:output:0*
element_dtype0*
_output_shapes
: G
add/yConst*
value	B :*
dtype0*
_output_shapes
: J
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_1AddV2while_loop_counteradd_1/y:output:0*
_output_shapes
: *
T0Z
IdentityIdentity	add_1:z:0^StatefulPartitionedCall*
T0*
_output_shapes
: k

Identity_1Identitywhile_maximum_iterations^StatefulPartitionedCall*
T0*
_output_shapes
: Z

Identity_2Identityadd:z:0^StatefulPartitionedCall*
_output_shapes
: *
T0�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^StatefulPartitionedCall*
T0*
_output_shapes
: �

Identity_4Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*'
_output_shapes
:��������� *
T0�

Identity_5Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� "P
%statefulpartitionedcall_lstm_1_kernel'statefulpartitionedcall_lstm_1_kernel_0"d
/statefulpartitionedcall_lstm_1_recurrent_kernel1statefulpartitionedcall_lstm_1_recurrent_kernel_0"L
#statefulpartitionedcall_lstm_1_bias%statefulpartitionedcall_lstm_1_bias_0"$
strided_slice_1strided_slice_1_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0*Q
_input_shapes@
>: : : : :��������� :��������� : : :::22
StatefulPartitionedCallStatefulPartitionedCall:  : : : : : : : : :	 :
 
�
�
%__inference_lstm_1_layer_call_fn_9696

inputs)
%statefulpartitionedcall_lstm_1_kernel'
#statefulpartitionedcall_lstm_1_bias3
/statefulpartitionedcall_lstm_1_recurrent_kernel
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs%statefulpartitionedcall_lstm_1_kernel#statefulpartitionedcall_lstm_1_bias/statefulpartitionedcall_lstm_1_recurrent_kernel*+
_gradient_op_typePartitionedCall-5947*I
fDRB
@__inference_lstm_1_layer_call_and_return_conditional_losses_5935*
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
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:��������� *
T0"
identityIdentity:output:0*6
_input_shapes%
#:���������(@:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
�
�
'__inference_features_layer_call_fn_7400

inputs/
+statefulpartitionedcall_features_embeddings
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs+statefulpartitionedcall_features_embeddings*K
fFRD
B__inference_features_layer_call_and_return_conditional_losses_4781*
Tout
2**
config_proto

GPU 

CPU2J 8*+
_output_shapes
:���������(*
Tin
2*+
_gradient_op_typePartitionedCall-4788�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������("
identityIdentity:output:0**
_input_shapes
:���������(:22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: 
��
�
>__inference_lstm_layer_call_and_return_conditional_losses_7958
inputs_0$
 split_readvariableop_lstm_kernel$
 split_1_readvariableop_lstm_bias(
$readvariableop_lstm_recurrent_kernel
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOp�while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
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
_output_shapes
: *
T0*
Index0*
shrink_axis_maskM
zeros/mul/yConst*
dtype0*
_output_shapes
: *
value	B :@_
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
value	B :@*
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
:���������@*
T0O
zeros_1/mul/yConst*
_output_shapes
: *
value	B :@*
dtype0c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
_output_shapes
: *
T0Q
zeros_1/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
_output_shapes
:*
T0R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:x
	transpose	Transposeinputs_0transpose/perm:output:0*4
_output_shapes"
 :������������������*
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
_output_shapes
: *
T0*
Index0*
shrink_axis_maskf
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
strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:a
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
Index0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: v
split/ReadVariableOpReadVariableOp split_readvariableop_lstm_kernel*
_output_shapes
:	�*
dtype0�
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
	num_split*<
_output_shapes*
(:@:@:@:@*
T0l
MatMulMatMulstrided_slice_2:output:0split:output:0*
T0*'
_output_shapes
:���������@n
MatMul_1MatMulstrided_slice_2:output:0split:output:1*
T0*'
_output_shapes
:���������@n
MatMul_2MatMulstrided_slice_2:output:0split:output:2*
T0*'
_output_shapes
:���������@n
MatMul_3MatMulstrided_slice_2:output:0split:output:3*
T0*'
_output_shapes
:���������@I
Const_1Const*
dtype0*
_output_shapes
: *
value	B :S
split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: t
split_1/ReadVariableOpReadVariableOp split_1_readvariableop_lstm_bias*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split*,
_output_shapes
:@:@:@:@h
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:���������@l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:���������@l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:���������@l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:���������@t
ReadVariableOpReadVariableOp$readvariableop_lstm_recurrent_kernel*
_output_shapes
:	@�*
dtype0f
strided_slice_3/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_3/stack_1Const*
_output_shapes
:*
valueB"    @   *
dtype0h
strided_slice_3/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_3StridedSliceReadVariableOp:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:@@*
Index0*
T0n
MatMul_4MatMulzeros:output:0strided_slice_3:output:0*
T0*'
_output_shapes
:���������@d
addAddV2BiasAdd:output:0MatMul_4:product:0*'
_output_shapes
:���������@*
T0L
Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *��L>L
Const_3Const*
dtype0*
_output_shapes
: *
valueB
 *   ?W
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:���������@Y
Add_1AddMul:z:0Const_3:output:0*'
_output_shapes
:���������@*
T0\
clip_by_value/Minimum/yConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*'
_output_shapes
:���������@*
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
:���������@�
ReadVariableOp_1ReadVariableOp$readvariableop_lstm_recurrent_kernel^ReadVariableOp*
dtype0*
_output_shapes
:	@�f
strided_slice_4/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_4/stack_1Const*
valueB"    �   *
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

:@@*
T0*
Index0n
MatMul_5MatMulzeros:output:0strided_slice_4:output:0*
T0*'
_output_shapes
:���������@h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*'
_output_shapes
:���������@*
T0L
Const_4Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_4:output:0*'
_output_shapes
:���������@*
T0[
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*'
_output_shapes
:���������@^
clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*'
_output_shapes
:���������@*
T0V
clip_by_value_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *    �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*'
_output_shapes
:���������@*
T0e
mul_2Mulclip_by_value_1:z:0zeros_1:output:0*
T0*'
_output_shapes
:���������@�
ReadVariableOp_2ReadVariableOp$readvariableop_lstm_recurrent_kernel^ReadVariableOp_1*
dtype0*
_output_shapes
:	@�f
strided_slice_5/stackConst*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_5/stack_1Const*
valueB"    �   *
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
end_mask*
_output_shapes

:@@n
MatMul_6MatMulzeros:output:0strided_slice_5:output:0*'
_output_shapes
:���������@*
T0h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*'
_output_shapes
:���������@*
T0I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:���������@[
mul_3Mulclip_by_value:z:0Tanh:y:0*'
_output_shapes
:���������@*
T0V
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:���������@�
ReadVariableOp_3ReadVariableOp$readvariableop_lstm_recurrent_kernel^ReadVariableOp_2*
dtype0*
_output_shapes
:	@�f
strided_slice_6/stackConst*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_6/stack_1Const*
_output_shapes
:*
valueB"        *
dtype0h
strided_slice_6/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_6StridedSliceReadVariableOp_3:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:@@*
T0*
Index0n
MatMul_7MatMulzeros:output:0strided_slice_6:output:0*'
_output_shapes
:���������@*
T0h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*'
_output_shapes
:���������@*
T0L
Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_4Mul	add_6:z:0Const_6:output:0*'
_output_shapes
:���������@*
T0[
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:���������@^
clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:���������@V
clip_by_value_2/yConst*
_output_shapes
: *
valueB
 *    *
dtype0�
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*'
_output_shapes
:���������@*
T0K
Tanh_1Tanh	add_5:z:0*'
_output_shapes
:���������@*
T0_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:���������@n
TensorArrayV2_1/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"����@   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
_output_shapes
: *
value	B : *
dtype0c
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 split_readvariableop_lstm_kernel split_1_readvariableop_lstm_bias$readvariableop_lstm_recurrent_kernel^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
parallel_iterations *
condR
while_cond_7804*
_num_original_outputs*
bodyR
while_body_7805*L
_output_shapes:
8: : : : :���������@:���������@: : : : : *
T
2*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
_lower_using_switch_merge(K
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
while/Identity_4Identitywhile:output:4*'
_output_shapes
:���������@*
T0^
while/Identity_5Identitywhile:output:5*'
_output_shapes
:���������@*
T0M
while/Identity_6Identitywhile:output:6*
_output_shapes
: *
T0M
while/Identity_7Identitywhile:output:7*
_output_shapes
: *
T0M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"����@   *
dtype0*
_output_shapes
:�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :������������������@h
strided_slice_7/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:a
strided_slice_7/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_7/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
strided_slice_7StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*'
_output_shapes
:���������@e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������@�
IdentityIdentitytranspose_1:y:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp^while*4
_output_shapes"
 :������������������@*
T0"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp2
whilewhile2 
ReadVariableOpReadVariableOp:( $
"
_user_specified_name
inputs/0: : : 
�
_
&__inference_dropout_layer_call_fn_9744

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*+
_gradient_op_typePartitionedCall-6020*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_6008*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:���������*
Tin
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:���������*
T0"
identityIdentity:output:0*&
_input_shapes
:���������22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
#__inference_lstm_layer_call_fn_8540

inputs'
#statefulpartitionedcall_lstm_kernel%
!statefulpartitionedcall_lstm_bias1
-statefulpartitionedcall_lstm_recurrent_kernel
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs#statefulpartitionedcall_lstm_kernel!statefulpartitionedcall_lstm_bias-statefulpartitionedcall_lstm_recurrent_kernel*+
_gradient_op_typePartitionedCall-5358*G
fBR@
>__inference_lstm_layer_call_and_return_conditional_losses_5076*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:���������(@�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*+
_output_shapes
:���������(@*
T0"
identityIdentity:output:0*6
_input_shapes%
#:���������(:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
�
�
while_cond_8673
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
lstm_1_kernel
lstm_1_bias
lstm_1_recurrent_kernel
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
_output_shapes
: *
T0
"
identityIdentity:output:0*Q
_input_shapes@
>: : : : :��������� :��������� : : :::: : : : : :	 :
 :  : : : 
�`
�
while_body_5202
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0&
"split_readvariableop_lstm_kernel_0&
"split_1_readvariableop_lstm_bias_0*
&readvariableop_lstm_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor$
 split_readvariableop_lstm_kernel$
 split_1_readvariableop_lstm_bias(
$readvariableop_lstm_recurrent_kernel��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
valueB"����   *
dtype0�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:���������G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: x
split/ReadVariableOpReadVariableOp"split_readvariableop_lstm_kernel_0*
dtype0*
_output_shapes
:	��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split*<
_output_shapes*
(:@:@:@:@~
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:0*'
_output_shapes
:���������@*
T0�
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:1*
T0*'
_output_shapes
:���������@�
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:2*
T0*'
_output_shapes
:���������@�
MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:3*
T0*'
_output_shapes
:���������@I
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: v
split_1/ReadVariableOpReadVariableOp"split_1_readvariableop_lstm_bias_0*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split*,
_output_shapes
:@:@:@:@h
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:���������@l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:���������@l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*'
_output_shapes
:���������@*
T0l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*'
_output_shapes
:���������@*
T0v
ReadVariableOpReadVariableOp&readvariableop_lstm_recurrent_kernel_0*
_output_shapes
:	@�*
dtype0d
strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB"    @   f
strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:@@*
T0*
Index0k
MatMul_4MatMulplaceholder_2strided_slice:output:0*
T0*'
_output_shapes
:���������@d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:���������@L
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
: W
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:���������@Y
Add_1AddMul:z:0Const_3:output:0*'
_output_shapes
:���������@*
T0\
clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������@T
clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������@�
ReadVariableOp_1ReadVariableOp&readvariableop_lstm_recurrent_kernel_0^ReadVariableOp*
_output_shapes
:	@�*
dtype0f
strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB"    @   h
strided_slice_2/stack_1Const*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_slice_2StridedSliceReadVariableOp_1:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:@@*
T0*
Index0m
MatMul_5MatMulplaceholder_2strided_slice_2:output:0*
T0*'
_output_shapes
:���������@h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:���������@L
Const_4Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_4:output:0*'
_output_shapes
:���������@*
T0[
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*'
_output_shapes
:���������@^
clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:���������@V
clip_by_value_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*'
_output_shapes
:���������@*
T0b
mul_2Mulclip_by_value_1:z:0placeholder_3*
T0*'
_output_shapes
:���������@�
ReadVariableOp_2ReadVariableOp&readvariableop_lstm_recurrent_kernel_0^ReadVariableOp_1*
dtype0*
_output_shapes
:	@�f
strided_slice_3/stackConst*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_3/stack_1Const*
valueB"    �   *
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
end_mask*
_output_shapes

:@@*
Index0*
T0m
MatMul_6MatMulplaceholder_2strided_slice_3:output:0*'
_output_shapes
:���������@*
T0h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*'
_output_shapes
:���������@*
T0I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:���������@[
mul_3Mulclip_by_value:z:0Tanh:y:0*'
_output_shapes
:���������@*
T0V
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:���������@�
ReadVariableOp_3ReadVariableOp&readvariableop_lstm_recurrent_kernel_0^ReadVariableOp_2*
dtype0*
_output_shapes
:	@�f
strided_slice_4/stackConst*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_4/stack_1Const*
_output_shapes
:*
valueB"        *
dtype0h
strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_4StridedSliceReadVariableOp_3:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:@@*
T0*
Index0m
MatMul_7MatMulplaceholder_2strided_slice_4:output:0*'
_output_shapes
:���������@*
T0h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*'
_output_shapes
:���������@*
T0L
Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_4Mul	add_6:z:0Const_6:output:0*'
_output_shapes
:���������@*
T0[
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:���������@^
clip_by_value_2/Minimum/yConst*
dtype0*
_output_shapes
: *
valueB
 *  �?�
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:���������@V
clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:���������@K
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:���������@_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:���������@�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_5:z:0*
element_dtype0*
_output_shapes
: I
add_8/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_8AddV2placeholderadd_8/y:output:0*
_output_shapes
: *
T0I
add_9/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_9AddV2while_loop_counteradd_9/y:output:0*
T0*
_output_shapes
: �
IdentityIdentity	add_9:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_1Identitywhile_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_2Identity	add_8:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_4Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:���������@�

Identity_5Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:���������@"F
 split_1_readvariableop_lstm_bias"split_1_readvariableop_lstm_bias_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"N
$readvariableop_lstm_recurrent_kernel&readvariableop_lstm_recurrent_kernel_0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"F
 split_readvariableop_lstm_kernel"split_readvariableop_lstm_kernel_0"$
strided_slice_1strided_slice_1_0*Q
_input_shapes@
>: : : : :���������@:���������@: : :::2 
ReadVariableOpReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp: :	 :
 :  : : : : : : : 
�b
�
!sequential_lstm_1_while_body_3105(
$sequential_lstm_1_while_loop_counter.
*sequential_lstm_1_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3'
#sequential_lstm_1_strided_slice_1_0c
_tensorarrayv2read_tensorlistgetitem_sequential_lstm_1_tensorarrayunstack_tensorlistfromtensor_0(
$split_readvariableop_lstm_1_kernel_0(
$split_1_readvariableop_lstm_1_bias_0,
(readvariableop_lstm_1_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5%
!sequential_lstm_1_strided_slice_1a
]tensorarrayv2read_tensorlistgetitem_sequential_lstm_1_tensorarrayunstack_tensorlistfromtensor&
"split_readvariableop_lstm_1_kernel&
"split_1_readvariableop_lstm_1_bias*
&readvariableop_lstm_1_recurrent_kernel��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����@   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItem_tensorarrayv2read_tensorlistgetitem_sequential_lstm_1_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:���������@G
ConstConst*
_output_shapes
: *
value	B :*
dtype0Q
split/split_dimConst*
_output_shapes
: *
value	B :*
dtype0z
split/ReadVariableOpReadVariableOp$split_readvariableop_lstm_1_kernel_0*
dtype0*
_output_shapes
:	@��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split*<
_output_shapes*
(:@ :@ :@ :@ ~
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:0*
T0*'
_output_shapes
:��������� �
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:1*'
_output_shapes
:��������� *
T0�
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:2*
T0*'
_output_shapes
:��������� �
MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:3*
T0*'
_output_shapes
:��������� I
Const_1Const*
dtype0*
_output_shapes
: *
value	B :S
split_1/split_dimConst*
dtype0*
_output_shapes
: *
value	B : x
split_1/ReadVariableOpReadVariableOp$split_1_readvariableop_lstm_1_bias_0*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split*,
_output_shapes
: : : : h
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:��������� l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*'
_output_shapes
:��������� *
T0l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:��������� l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*'
_output_shapes
:��������� *
T0x
ReadVariableOpReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0*
dtype0*
_output_shapes
:	 �d
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
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
Index0*
T0k
MatMul_4MatMulplaceholder_2strided_slice:output:0*'
_output_shapes
:��������� *
T0d
addAddV2BiasAdd:output:0MatMul_4:product:0*
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
: W
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:��������� Y
Add_1AddMul:z:0Const_3:output:0*
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
clip_by_value/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*'
_output_shapes
:��������� *
T0�
ReadVariableOp_1ReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0^ReadVariableOp*
dtype0*
_output_shapes
:	 �f
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
dtype0*
_output_shapes
:*
valueB"      �
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
Index0*
T0m
MatMul_5MatMulplaceholder_2strided_slice_1:output:0*
T0*'
_output_shapes
:��������� h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:��������� L
Const_4Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:��������� [
Add_3Add	Mul_1:z:0Const_5:output:0*
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
:��������� b
mul_2Mulclip_by_value_1:z:0placeholder_3*
T0*'
_output_shapes
:��������� �
ReadVariableOp_2ReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0^ReadVariableOp_1*
dtype0*
_output_shapes
:	 �f
strided_slice_2/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_1Const*
valueB"    `   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:  m
MatMul_6MatMulplaceholder_2strided_slice_2:output:0*'
_output_shapes
:��������� *
T0h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*'
_output_shapes
:��������� *
T0I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:��������� [
mul_3Mulclip_by_value:z:0Tanh:y:0*'
_output_shapes
:��������� *
T0V
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_3ReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0^ReadVariableOp_2*
dtype0*
_output_shapes
:	 �f
strided_slice_3/stackConst*
valueB"    `   *
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
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
T0*
Index0m
MatMul_7MatMulplaceholder_2strided_slice_3:output:0*
T0*'
_output_shapes
:��������� h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:��������� L
Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_4Mul	add_6:z:0Const_6:output:0*'
_output_shapes
:��������� *
T0[
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:��������� ^
clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*'
_output_shapes
:��������� *
T0V
clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:��������� K
Tanh_1Tanh	add_5:z:0*'
_output_shapes
:��������� *
T0_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:��������� �
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_5:z:0*
element_dtype0*
_output_shapes
: I
add_8/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_8AddV2placeholderadd_8/y:output:0*
T0*
_output_shapes
: I
add_9/yConst*
_output_shapes
: *
value	B :*
dtype0g
add_9AddV2$sequential_lstm_1_while_loop_counteradd_9/y:output:0*
_output_shapes
: *
T0�
IdentityIdentity	add_9:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_1Identity*sequential_lstm_1_while_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_2Identity	add_8:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_4Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:��������� �

Identity_5Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:��������� "!

identity_4Identity_4:output:0"J
"split_1_readvariableop_lstm_1_bias$split_1_readvariableop_lstm_1_bias_0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"H
!sequential_lstm_1_strided_slice_1#sequential_lstm_1_strided_slice_1_0"J
"split_readvariableop_lstm_1_kernel$split_readvariableop_lstm_1_kernel_0"�
]tensorarrayv2read_tensorlistgetitem_sequential_lstm_1_tensorarrayunstack_tensorlistfromtensor_tensorarrayv2read_tensorlistgetitem_sequential_lstm_1_tensorarrayunstack_tensorlistfromtensor_0"R
&readvariableop_lstm_1_recurrent_kernel(readvariableop_lstm_1_recurrent_kernel_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*Q
_input_shapes@
>: : : : :��������� :��������� : : :::20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp2 
ReadVariableOpReadVariableOp: : : : : : : : :	 :
 :  
�
�
while_cond_5201
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
lstm_kernel
	lstm_bias
lstm_recurrent_kernel
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
identityIdentity:output:0*Q
_input_shapes@
>: : : : :���������@:���������@: : :::: : : :	 :
 :  : : : : : 
� 
�
D__inference_sequential_layer_call_and_return_conditional_losses_6154

inputs8
4features_statefulpartitionedcall_features_embeddings,
(lstm_statefulpartitionedcall_lstm_kernel*
&lstm_statefulpartitionedcall_lstm_bias6
2lstm_statefulpartitionedcall_lstm_recurrent_kernel0
,lstm_1_statefulpartitionedcall_lstm_1_kernel.
*lstm_1_statefulpartitionedcall_lstm_1_bias:
6lstm_1_statefulpartitionedcall_lstm_1_recurrent_kernel.
*dense_statefulpartitionedcall_dense_kernel,
(dense_statefulpartitionedcall_dense_bias2
.dense_1_statefulpartitionedcall_dense_1_kernel0
,dense_1_statefulpartitionedcall_dense_1_bias
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall� features/StatefulPartitionedCall�lstm/StatefulPartitionedCall�lstm_1/StatefulPartitionedCall�
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
:���������(*+
_gradient_op_typePartitionedCall-4788*K
fFRD
B__inference_features_layer_call_and_return_conditional_losses_4781�
lstm/StatefulPartitionedCallStatefulPartitionedCall)features/StatefulPartitionedCall:output:0(lstm_statefulpartitionedcall_lstm_kernel&lstm_statefulpartitionedcall_lstm_bias2lstm_statefulpartitionedcall_lstm_recurrent_kernel*+
_gradient_op_typePartitionedCall-5367*G
fBR@
>__inference_lstm_layer_call_and_return_conditional_losses_5355*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:���������(@�
lstm_1/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0,lstm_1_statefulpartitionedcall_lstm_1_kernel*lstm_1_statefulpartitionedcall_lstm_1_bias6lstm_1_statefulpartitionedcall_lstm_1_recurrent_kernel*
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
_gradient_op_typePartitionedCall-5947*I
fDRB
@__inference_lstm_1_layer_call_and_return_conditional_losses_5935�
dense/StatefulPartitionedCallStatefulPartitionedCall'lstm_1/StatefulPartitionedCall:output:0*dense_statefulpartitionedcall_dense_kernel(dense_statefulpartitionedcall_dense_bias*
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
2*+
_gradient_op_typePartitionedCall-5976*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_5969�
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:���������*+
_gradient_op_typePartitionedCall-6029*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_6016�
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0.dense_1_statefulpartitionedcall_dense_1_kernel,dense_1_statefulpartitionedcall_dense_1_bias*'
_output_shapes
:���������*
Tin
2*+
_gradient_op_typePartitionedCall-6052*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_6045*
Tout
2**
config_proto

GPU 

CPU2J 8�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall!^features/StatefulPartitionedCall^lstm/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*R
_input_shapesA
?:���������(:::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 features/StatefulPartitionedCall features/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall: : : : : : :	 :
 : :& "
 
_user_specified_nameinputs: : 
��
�
!__inference__traced_restore_10465
file_prefix(
$assignvariableop_features_embeddings#
assignvariableop_1_dense_kernel!
assignvariableop_2_dense_bias%
!assignvariableop_3_dense_1_kernel#
assignvariableop_4_dense_1_bias)
%assignvariableop_5_training_adam_iter+
'assignvariableop_6_training_adam_beta_1+
'assignvariableop_7_training_adam_beta_2*
&assignvariableop_8_training_adam_decay2
.assignvariableop_9_training_adam_learning_rate#
assignvariableop_10_lstm_kernel-
)assignvariableop_11_lstm_recurrent_kernel!
assignvariableop_12_lstm_bias%
!assignvariableop_13_lstm_1_kernel/
+assignvariableop_14_lstm_1_recurrent_kernel#
assignvariableop_15_lstm_1_bias
assignvariableop_16_total
assignvariableop_17_count_3;
7assignvariableop_18_training_adam_features_embeddings_m4
0assignvariableop_19_training_adam_dense_kernel_m2
.assignvariableop_20_training_adam_dense_bias_m6
2assignvariableop_21_training_adam_dense_1_kernel_m4
0assignvariableop_22_training_adam_dense_1_bias_m3
/assignvariableop_23_training_adam_lstm_kernel_m=
9assignvariableop_24_training_adam_lstm_recurrent_kernel_m1
-assignvariableop_25_training_adam_lstm_bias_m5
1assignvariableop_26_training_adam_lstm_1_kernel_m?
;assignvariableop_27_training_adam_lstm_1_recurrent_kernel_m3
/assignvariableop_28_training_adam_lstm_1_bias_m;
7assignvariableop_29_training_adam_features_embeddings_v4
0assignvariableop_30_training_adam_dense_kernel_v2
.assignvariableop_31_training_adam_dense_bias_v6
2assignvariableop_32_training_adam_dense_1_kernel_v4
0assignvariableop_33_training_adam_dense_1_bias_v3
/assignvariableop_34_training_adam_lstm_kernel_v=
9assignvariableop_35_training_adam_lstm_recurrent_kernel_v1
-assignvariableop_36_training_adam_lstm_bias_v5
1assignvariableop_37_training_adam_lstm_1_kernel_v?
;assignvariableop_38_training_adam_lstm_1_recurrent_kernel_v3
/assignvariableop_39_training_adam_lstm_1_bias_v
identity_41��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�(B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:(�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*6
dtypes,
*2(	*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp$assignvariableop_features_embeddingsIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
_output_shapes
:*
T0
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_kernelIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
_output_shapes
:*
T0}
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_biasIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_1_kernelIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_1_biasIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
_output_shapes
:*
T0	�
AssignVariableOp_5AssignVariableOp%assignvariableop_5_training_adam_iterIdentity_5:output:0*
dtype0	*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp'assignvariableop_6_training_adam_beta_1Identity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
_output_shapes
:*
T0�
AssignVariableOp_7AssignVariableOp'assignvariableop_7_training_adam_beta_2Identity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp&assignvariableop_8_training_adam_decayIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
_output_shapes
:*
T0�
AssignVariableOp_9AssignVariableOp.assignvariableop_9_training_adam_learning_rateIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_lstm_kernelIdentity_10:output:0*
_output_shapes
 *
dtype0P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp)assignvariableop_11_lstm_recurrent_kernelIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
_output_shapes
:*
T0
AssignVariableOp_12AssignVariableOpassignvariableop_12_lstm_biasIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
_output_shapes
:*
T0�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_lstm_1_kernelIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
_output_shapes
:*
T0�
AssignVariableOp_14AssignVariableOp+assignvariableop_14_lstm_1_recurrent_kernelIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
_output_shapes
:*
T0�
AssignVariableOp_15AssignVariableOpassignvariableop_15_lstm_1_biasIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:{
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0*
_output_shapes
 *
dtype0P
Identity_17IdentityRestoreV2:tensors:17*
_output_shapes
:*
T0}
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_3Identity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
_output_shapes
:*
T0�
AssignVariableOp_18AssignVariableOp7assignvariableop_18_training_adam_features_embeddings_mIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp0assignvariableop_19_training_adam_dense_kernel_mIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp.assignvariableop_20_training_adam_dense_bias_mIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp2assignvariableop_21_training_adam_dense_1_kernel_mIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp0assignvariableop_22_training_adam_dense_1_bias_mIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
_output_shapes
:*
T0�
AssignVariableOp_23AssignVariableOp/assignvariableop_23_training_adam_lstm_kernel_mIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp9assignvariableop_24_training_adam_lstm_recurrent_kernel_mIdentity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp-assignvariableop_25_training_adam_lstm_bias_mIdentity_25:output:0*
dtype0*
_output_shapes
 P
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp1assignvariableop_26_training_adam_lstm_1_kernel_mIdentity_26:output:0*
dtype0*
_output_shapes
 P
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp;assignvariableop_27_training_adam_lstm_1_recurrent_kernel_mIdentity_27:output:0*
dtype0*
_output_shapes
 P
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp/assignvariableop_28_training_adam_lstm_1_bias_mIdentity_28:output:0*
dtype0*
_output_shapes
 P
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp7assignvariableop_29_training_adam_features_embeddings_vIdentity_29:output:0*
dtype0*
_output_shapes
 P
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp0assignvariableop_30_training_adam_dense_kernel_vIdentity_30:output:0*
dtype0*
_output_shapes
 P
Identity_31IdentityRestoreV2:tensors:31*
_output_shapes
:*
T0�
AssignVariableOp_31AssignVariableOp.assignvariableop_31_training_adam_dense_bias_vIdentity_31:output:0*
dtype0*
_output_shapes
 P
Identity_32IdentityRestoreV2:tensors:32*
_output_shapes
:*
T0�
AssignVariableOp_32AssignVariableOp2assignvariableop_32_training_adam_dense_1_kernel_vIdentity_32:output:0*
dtype0*
_output_shapes
 P
Identity_33IdentityRestoreV2:tensors:33*
_output_shapes
:*
T0�
AssignVariableOp_33AssignVariableOp0assignvariableop_33_training_adam_dense_1_bias_vIdentity_33:output:0*
dtype0*
_output_shapes
 P
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp/assignvariableop_34_training_adam_lstm_kernel_vIdentity_34:output:0*
dtype0*
_output_shapes
 P
Identity_35IdentityRestoreV2:tensors:35*
_output_shapes
:*
T0�
AssignVariableOp_35AssignVariableOp9assignvariableop_35_training_adam_lstm_recurrent_kernel_vIdentity_35:output:0*
_output_shapes
 *
dtype0P
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp-assignvariableop_36_training_adam_lstm_bias_vIdentity_36:output:0*
dtype0*
_output_shapes
 P
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp1assignvariableop_37_training_adam_lstm_1_kernel_vIdentity_37:output:0*
dtype0*
_output_shapes
 P
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp;assignvariableop_38_training_adam_lstm_1_recurrent_kernel_vIdentity_38:output:0*
_output_shapes
 *
dtype0P
Identity_39IdentityRestoreV2:tensors:39*
_output_shapes
:*
T0�
AssignVariableOp_39AssignVariableOp/assignvariableop_39_training_adam_lstm_1_bias_vIdentity_39:output:0*
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
 �
Identity_40Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: �
Identity_41IdentityIdentity_40:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_41Identity_41:output:0*�
_input_shapes�
�: ::::::::::::::::::::::::::::::::::::::::2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112
RestoreV2_1RestoreV2_12*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_31AssignVariableOp_312$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_29AssignVariableOp_292*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV2: : : : : : : : : : :  :! :" :# :$ :% :& :' :( :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : 
�a
�
lstm_1_while_body_6596
lstm_1_while_loop_counter#
lstm_1_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
lstm_1_strided_slice_1_0X
Ttensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0(
$split_readvariableop_lstm_1_kernel_0(
$split_1_readvariableop_lstm_1_bias_0,
(readvariableop_lstm_1_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
lstm_1_strided_slice_1V
Rtensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor&
"split_readvariableop_lstm_1_kernel&
"split_1_readvariableop_lstm_1_bias*
&readvariableop_lstm_1_recurrent_kernel��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����@   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemTtensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:���������@G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: z
split/ReadVariableOpReadVariableOp$split_readvariableop_lstm_1_kernel_0*
dtype0*
_output_shapes
:	@��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split*<
_output_shapes*
(:@ :@ :@ :@ ~
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:0*
T0*'
_output_shapes
:��������� �
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:1*
T0*'
_output_shapes
:��������� �
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:2*
T0*'
_output_shapes
:��������� �
MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:3*
T0*'
_output_shapes
:��������� I
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
dtype0*
_output_shapes
: *
value	B : x
split_1/ReadVariableOpReadVariableOp$split_1_readvariableop_lstm_1_bias_0*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split*,
_output_shapes
: : : : h
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:��������� l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:��������� l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:��������� l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*'
_output_shapes
:��������� *
T0x
ReadVariableOpReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0*
dtype0*
_output_shapes
:	 �d
strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB"        f
strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
Index0*
T0k
MatMul_4MatMulplaceholder_2strided_slice:output:0*
T0*'
_output_shapes
:��������� d
addAddV2BiasAdd:output:0MatMul_4:product:0*
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
: W
MulMuladd:z:0Const_2:output:0*'
_output_shapes
:��������� *
T0Y
Add_1AddMul:z:0Const_3:output:0*
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
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*'
_output_shapes
:��������� *
T0�
ReadVariableOp_1ReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0^ReadVariableOp*
dtype0*
_output_shapes
:	 �f
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

:  *
T0*
Index0m
MatMul_5MatMulplaceholder_2strided_slice_1:output:0*
T0*'
_output_shapes
:��������� h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:��������� L
Const_4Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_5Const*
dtype0*
_output_shapes
: *
valueB
 *   ?[
Mul_1Mul	add_2:z:0Const_4:output:0*'
_output_shapes
:��������� *
T0[
Add_3Add	Mul_1:z:0Const_5:output:0*
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
mul_2Mulclip_by_value_1:z:0placeholder_3*'
_output_shapes
:��������� *
T0�
ReadVariableOp_2ReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0^ReadVariableOp_1*
dtype0*
_output_shapes
:	 �f
strided_slice_2/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_1Const*
valueB"    `   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
Index0*
T0m
MatMul_6MatMulplaceholder_2strided_slice_2:output:0*
T0*'
_output_shapes
:��������� h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*'
_output_shapes
:��������� *
T0I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:��������� [
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_5AddV2	mul_2:z:0	mul_3:z:0*'
_output_shapes
:��������� *
T0�
ReadVariableOp_3ReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0^ReadVariableOp_2*
_output_shapes
:	 �*
dtype0f
strided_slice_3/stackConst*
valueB"    `   *
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
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:  m
MatMul_7MatMulplaceholder_2strided_slice_3:output:0*
T0*'
_output_shapes
:��������� h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:��������� L
Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_4Mul	add_6:z:0Const_6:output:0*'
_output_shapes
:��������� *
T0[
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:��������� ^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
valueB
 *  �?*
dtype0�
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*'
_output_shapes
:��������� *
T0V
clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*'
_output_shapes
:��������� *
T0K
Tanh_1Tanh	add_5:z:0*'
_output_shapes
:��������� *
T0_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*'
_output_shapes
:��������� *
T0�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_5:z:0*
element_dtype0*
_output_shapes
: I
add_8/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_8AddV2placeholderadd_8/y:output:0*
T0*
_output_shapes
: I
add_9/yConst*
dtype0*
_output_shapes
: *
value	B :\
add_9AddV2lstm_1_while_loop_counteradd_9/y:output:0*
T0*
_output_shapes
: �
IdentityIdentity	add_9:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_1Identitylstm_1_while_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_2Identity	add_8:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_4Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:��������� �

Identity_5Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*'
_output_shapes
:��������� *
T0"J
"split_readvariableop_lstm_1_kernel$split_readvariableop_lstm_1_kernel_0"R
&readvariableop_lstm_1_recurrent_kernel(readvariableop_lstm_1_recurrent_kernel_0"�
Rtensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensorTtensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"2
lstm_1_strided_slice_1lstm_1_strided_slice_1_0"J
"split_1_readvariableop_lstm_1_bias$split_1_readvariableop_lstm_1_bias_0"!

identity_5Identity_5:output:0"
identityIdentity:output:0*Q
_input_shapes@
>: : : : :��������� :��������� : : :::2 
ReadVariableOpReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp: : : :	 :
 :  : : : : : 
�P
�
__inference__traced_save_10332
file_prefix2
.savev2_features_embeddings_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop1
-savev2_training_adam_iter_read_readvariableop	3
/savev2_training_adam_beta_1_read_readvariableop3
/savev2_training_adam_beta_2_read_readvariableop2
.savev2_training_adam_decay_read_readvariableop:
6savev2_training_adam_learning_rate_read_readvariableop*
&savev2_lstm_kernel_read_readvariableop4
0savev2_lstm_recurrent_kernel_read_readvariableop(
$savev2_lstm_bias_read_readvariableop,
(savev2_lstm_1_kernel_read_readvariableop6
2savev2_lstm_1_recurrent_kernel_read_readvariableop*
&savev2_lstm_1_bias_read_readvariableop$
 savev2_total_read_readvariableop&
"savev2_count_3_read_readvariableopB
>savev2_training_adam_features_embeddings_m_read_readvariableop;
7savev2_training_adam_dense_kernel_m_read_readvariableop9
5savev2_training_adam_dense_bias_m_read_readvariableop=
9savev2_training_adam_dense_1_kernel_m_read_readvariableop;
7savev2_training_adam_dense_1_bias_m_read_readvariableop:
6savev2_training_adam_lstm_kernel_m_read_readvariableopD
@savev2_training_adam_lstm_recurrent_kernel_m_read_readvariableop8
4savev2_training_adam_lstm_bias_m_read_readvariableop<
8savev2_training_adam_lstm_1_kernel_m_read_readvariableopF
Bsavev2_training_adam_lstm_1_recurrent_kernel_m_read_readvariableop:
6savev2_training_adam_lstm_1_bias_m_read_readvariableopB
>savev2_training_adam_features_embeddings_v_read_readvariableop;
7savev2_training_adam_dense_kernel_v_read_readvariableop9
5savev2_training_adam_dense_bias_v_read_readvariableop=
9savev2_training_adam_dense_1_kernel_v_read_readvariableop;
7savev2_training_adam_dense_1_bias_v_read_readvariableop:
6savev2_training_adam_lstm_kernel_v_read_readvariableopD
@savev2_training_adam_lstm_recurrent_kernel_v_read_readvariableop8
4savev2_training_adam_lstm_bias_v_read_readvariableop<
8savev2_training_adam_lstm_1_kernel_v_read_readvariableopF
Bsavev2_training_adam_lstm_1_recurrent_kernel_v_read_readvariableop:
6savev2_training_adam_lstm_1_bias_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_7e1b40750e90478fbceedc4d42414283/part*
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
: �
SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:(*�
value�B�(B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE�
SaveV2/shape_and_slicesConst"/device:CPU:0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_features_embeddings_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop-savev2_training_adam_iter_read_readvariableop/savev2_training_adam_beta_1_read_readvariableop/savev2_training_adam_beta_2_read_readvariableop.savev2_training_adam_decay_read_readvariableop6savev2_training_adam_learning_rate_read_readvariableop&savev2_lstm_kernel_read_readvariableop0savev2_lstm_recurrent_kernel_read_readvariableop$savev2_lstm_bias_read_readvariableop(savev2_lstm_1_kernel_read_readvariableop2savev2_lstm_1_recurrent_kernel_read_readvariableop&savev2_lstm_1_bias_read_readvariableop savev2_total_read_readvariableop"savev2_count_3_read_readvariableop>savev2_training_adam_features_embeddings_m_read_readvariableop7savev2_training_adam_dense_kernel_m_read_readvariableop5savev2_training_adam_dense_bias_m_read_readvariableop9savev2_training_adam_dense_1_kernel_m_read_readvariableop7savev2_training_adam_dense_1_bias_m_read_readvariableop6savev2_training_adam_lstm_kernel_m_read_readvariableop@savev2_training_adam_lstm_recurrent_kernel_m_read_readvariableop4savev2_training_adam_lstm_bias_m_read_readvariableop8savev2_training_adam_lstm_1_kernel_m_read_readvariableopBsavev2_training_adam_lstm_1_recurrent_kernel_m_read_readvariableop6savev2_training_adam_lstm_1_bias_m_read_readvariableop>savev2_training_adam_features_embeddings_v_read_readvariableop7savev2_training_adam_dense_kernel_v_read_readvariableop5savev2_training_adam_dense_bias_v_read_readvariableop9savev2_training_adam_dense_1_kernel_v_read_readvariableop7savev2_training_adam_dense_1_bias_v_read_readvariableop6savev2_training_adam_lstm_kernel_v_read_readvariableop@savev2_training_adam_lstm_recurrent_kernel_v_read_readvariableop4savev2_training_adam_lstm_bias_v_read_readvariableop8savev2_training_adam_lstm_1_kernel_v_read_readvariableopBsavev2_training_adam_lstm_1_recurrent_kernel_v_read_readvariableop6savev2_training_adam_lstm_1_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	h
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
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 �
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
_output_shapes
:*
T0�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
_output_shapes
: *
T0"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :
��: :::: : : : : :	�:	@�:�:	@�:	 �:�: : :
��: ::::	�:	@�:�:	@�:	 �:�:
��: ::::	�:	@�:�:	@�:	 �:�: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :' :( :) 
�
�
)__inference_sequential_layer_call_fn_7385

inputs/
+statefulpartitionedcall_features_embeddings'
#statefulpartitionedcall_lstm_kernel%
!statefulpartitionedcall_lstm_bias1
-statefulpartitionedcall_lstm_recurrent_kernel)
%statefulpartitionedcall_lstm_1_kernel'
#statefulpartitionedcall_lstm_1_bias3
/statefulpartitionedcall_lstm_1_recurrent_kernel(
$statefulpartitionedcall_dense_kernel&
"statefulpartitionedcall_dense_bias*
&statefulpartitionedcall_dense_1_kernel(
$statefulpartitionedcall_dense_1_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs+statefulpartitionedcall_features_embeddings#statefulpartitionedcall_lstm_kernel!statefulpartitionedcall_lstm_bias-statefulpartitionedcall_lstm_recurrent_kernel%statefulpartitionedcall_lstm_1_kernel#statefulpartitionedcall_lstm_1_bias/statefulpartitionedcall_lstm_1_recurrent_kernel$statefulpartitionedcall_dense_kernel"statefulpartitionedcall_dense_bias&statefulpartitionedcall_dense_1_kernel$statefulpartitionedcall_dense_1_bias*
Tin
2*'
_output_shapes
:���������*+
_gradient_op_typePartitionedCall-6155*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_6154*
Tout
2**
config_proto

GPU 

CPU2J 8�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*R
_input_shapesA
?:���������(:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : 
�
�
while_cond_5781
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
lstm_1_kernel
lstm_1_bias
lstm_1_recurrent_kernel
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
identityIdentity:output:0*Q
_input_shapes@
>: : : : :��������� :��������� : : :::: : : : : : :	 :
 :  : : 
��
�
>__inference_lstm_layer_call_and_return_conditional_losses_8532

inputs$
 split_readvariableop_lstm_kernel$
 split_1_readvariableop_lstm_bias(
$readvariableop_lstm_recurrent_kernel
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
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
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
_output_shapes
: *
T0O
zeros/Less/yConst*
dtype0*
_output_shapes
: *
value
B :�Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
_output_shapes
: *
T0P
zeros/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
_output_shapes
:*
T0P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@O
zeros_1/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
_output_shapes
: *
T0R
zeros_1/packed/1Const*
_output_shapes
: *
value	B :@*
dtype0w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*'
_output_shapes
:���������@*
T0c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:(���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:a
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
TensorArrayV2/element_shapeConst*
_output_shapes
: *
valueB :
���������*
dtype0�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
element_dtype0*
_output_shapes
: *

shape_type0�
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
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:���������*
Index0*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: v
split/ReadVariableOpReadVariableOp split_readvariableop_lstm_kernel*
dtype0*
_output_shapes
:	��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
	num_split*<
_output_shapes*
(:@:@:@:@*
T0l
MatMulMatMulstrided_slice_2:output:0split:output:0*
T0*'
_output_shapes
:���������@n
MatMul_1MatMulstrided_slice_2:output:0split:output:1*
T0*'
_output_shapes
:���������@n
MatMul_2MatMulstrided_slice_2:output:0split:output:2*
T0*'
_output_shapes
:���������@n
MatMul_3MatMulstrided_slice_2:output:0split:output:3*'
_output_shapes
:���������@*
T0I
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: t
split_1/ReadVariableOpReadVariableOp split_1_readvariableop_lstm_bias*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split*,
_output_shapes
:@:@:@:@h
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:���������@l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:���������@l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:���������@l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:���������@t
ReadVariableOpReadVariableOp$readvariableop_lstm_recurrent_kernel*
dtype0*
_output_shapes
:	@�f
strided_slice_3/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB"    @   h
strided_slice_3/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_3StridedSliceReadVariableOp:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:@@*
T0*
Index0n
MatMul_4MatMulzeros:output:0strided_slice_3:output:0*
T0*'
_output_shapes
:���������@d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:���������@L
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
: W
MulMuladd:z:0Const_2:output:0*'
_output_shapes
:���������@*
T0Y
Add_1AddMul:z:0Const_3:output:0*'
_output_shapes
:���������@*
T0\
clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������@T
clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������@�
ReadVariableOp_1ReadVariableOp$readvariableop_lstm_recurrent_kernel^ReadVariableOp*
dtype0*
_output_shapes
:	@�f
strided_slice_4/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_4/stack_1Const*
valueB"    �   *
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

:@@*
T0*
Index0n
MatMul_5MatMulzeros:output:0strided_slice_4:output:0*
T0*'
_output_shapes
:���������@h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*'
_output_shapes
:���������@*
T0L
Const_4Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:���������@[
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*'
_output_shapes
:���������@^
clip_by_value_1/Minimum/yConst*
dtype0*
_output_shapes
: *
valueB
 *  �?�
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*'
_output_shapes
:���������@*
T0V
clip_by_value_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *    �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*'
_output_shapes
:���������@*
T0e
mul_2Mulclip_by_value_1:z:0zeros_1:output:0*
T0*'
_output_shapes
:���������@�
ReadVariableOp_2ReadVariableOp$readvariableop_lstm_recurrent_kernel^ReadVariableOp_1*
dtype0*
_output_shapes
:	@�f
strided_slice_5/stackConst*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_5/stack_1Const*
dtype0*
_output_shapes
:*
valueB"    �   h
strided_slice_5/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_slice_5StridedSliceReadVariableOp_2:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:@@*
Index0*
T0n
MatMul_6MatMulzeros:output:0strided_slice_5:output:0*'
_output_shapes
:���������@*
T0h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*'
_output_shapes
:���������@*
T0I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:���������@[
mul_3Mulclip_by_value:z:0Tanh:y:0*'
_output_shapes
:���������@*
T0V
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:���������@�
ReadVariableOp_3ReadVariableOp$readvariableop_lstm_recurrent_kernel^ReadVariableOp_2*
dtype0*
_output_shapes
:	@�f
strided_slice_6/stackConst*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_6/stack_1Const*
dtype0*
_output_shapes
:*
valueB"        h
strided_slice_6/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_6StridedSliceReadVariableOp_3:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:@@n
MatMul_7MatMulzeros:output:0strided_slice_6:output:0*
T0*'
_output_shapes
:���������@h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:���������@L
Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:���������@[
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:���������@^
clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*'
_output_shapes
:���������@*
T0V
clip_by_value_2/yConst*
_output_shapes
: *
valueB
 *    *
dtype0�
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:���������@K
Tanh_1Tanh	add_5:z:0*'
_output_shapes
:���������@*
T0_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*'
_output_shapes
:���������@*
T0n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
valueB"����@   *
dtype0�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
_output_shapes
: *
value	B : *
dtype0c
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 split_readvariableop_lstm_kernel split_1_readvariableop_lstm_bias$readvariableop_lstm_recurrent_kernel^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
bodyR
while_body_8379*L
_output_shapes:
8: : : : :���������@:���������@: : : : : *K
output_shapes:
8: : : : :���������@:���������@: : : : : *
T
2*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_8378*
_num_original_outputsK
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
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:���������@^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:���������@M
while/Identity_6Identitywhile:output:6*
_output_shapes
: *
T0M
while/Identity_7Identitywhile:output:7*
_output_shapes
: *
T0M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
_output_shapes
: *
T0O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"����@   *
dtype0*
_output_shapes
:�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:(���������@h
strided_slice_7/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:a
strided_slice_7/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_7StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*'
_output_shapes
:���������@*
T0*
Index0*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*!
valueB"          *
dtype0�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������(@�
IdentityIdentitytranspose_1:y:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp^while*
T0*+
_output_shapes
:���������(@"
identityIdentity:output:0*6
_input_shapes%
#:���������(:::2
whilewhile2 
ReadVariableOpReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp:& "
 
_user_specified_nameinputs: : : 
�`
�
lstm_while_body_6910
lstm_while_loop_counter!
lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
lstm_strided_slice_1_0V
Rtensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0&
"split_readvariableop_lstm_kernel_0&
"split_1_readvariableop_lstm_bias_0*
&readvariableop_lstm_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
lstm_strided_slice_1T
Ptensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor$
 split_readvariableop_lstm_kernel$
 split_1_readvariableop_lstm_bias(
$readvariableop_lstm_recurrent_kernel��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemRtensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:���������G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: x
split/ReadVariableOpReadVariableOp"split_readvariableop_lstm_kernel_0*
dtype0*
_output_shapes
:	��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
	num_split*<
_output_shapes*
(:@:@:@:@*
T0~
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:0*'
_output_shapes
:���������@*
T0�
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:1*
T0*'
_output_shapes
:���������@�
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:2*
T0*'
_output_shapes
:���������@�
MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:3*
T0*'
_output_shapes
:���������@I
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: v
split_1/ReadVariableOpReadVariableOp"split_1_readvariableop_lstm_bias_0*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split*,
_output_shapes
:@:@:@:@h
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:���������@l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:���������@l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*'
_output_shapes
:���������@*
T0l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:���������@v
ReadVariableOpReadVariableOp&readvariableop_lstm_recurrent_kernel_0*
dtype0*
_output_shapes
:	@�d
strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB"    @   f
strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:@@*
T0*
Index0k
MatMul_4MatMulplaceholder_2strided_slice:output:0*
T0*'
_output_shapes
:���������@d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:���������@L
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
 *   ?W
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:���������@Y
Add_1AddMul:z:0Const_3:output:0*
T0*'
_output_shapes
:���������@\
clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������@T
clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������@�
ReadVariableOp_1ReadVariableOp&readvariableop_lstm_recurrent_kernel_0^ReadVariableOp*
dtype0*
_output_shapes
:	@�f
strided_slice_1/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB"    �   h
strided_slice_1/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:@@m
MatMul_5MatMulplaceholder_2strided_slice_1:output:0*
T0*'
_output_shapes
:���������@h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:���������@L
Const_4Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_5Const*
dtype0*
_output_shapes
: *
valueB
 *   ?[
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:���������@[
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*'
_output_shapes
:���������@^
clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:���������@V
clip_by_value_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *    �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*'
_output_shapes
:���������@*
T0b
mul_2Mulclip_by_value_1:z:0placeholder_3*
T0*'
_output_shapes
:���������@�
ReadVariableOp_2ReadVariableOp&readvariableop_lstm_recurrent_kernel_0^ReadVariableOp_1*
dtype0*
_output_shapes
:	@�f
strided_slice_2/stackConst*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_1Const*
valueB"    �   *
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
end_mask*
_output_shapes

:@@*
Index0*
T0m
MatMul_6MatMulplaceholder_2strided_slice_2:output:0*
T0*'
_output_shapes
:���������@h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:���������@I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:���������@[
mul_3Mulclip_by_value:z:0Tanh:y:0*'
_output_shapes
:���������@*
T0V
add_5AddV2	mul_2:z:0	mul_3:z:0*'
_output_shapes
:���������@*
T0�
ReadVariableOp_3ReadVariableOp&readvariableop_lstm_recurrent_kernel_0^ReadVariableOp_2*
_output_shapes
:	@�*
dtype0f
strided_slice_3/stackConst*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_3/stack_1Const*
_output_shapes
:*
valueB"        *
dtype0h
strided_slice_3/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:@@*
Index0*
T0m
MatMul_7MatMulplaceholder_2strided_slice_3:output:0*'
_output_shapes
:���������@*
T0h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*'
_output_shapes
:���������@*
T0L
Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:���������@[
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:���������@^
clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:���������@V
clip_by_value_2/yConst*
dtype0*
_output_shapes
: *
valueB
 *    �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:���������@K
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:���������@_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:���������@�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_5:z:0*
element_dtype0*
_output_shapes
: I
add_8/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_8AddV2placeholderadd_8/y:output:0*
_output_shapes
: *
T0I
add_9/yConst*
_output_shapes
: *
value	B :*
dtype0Z
add_9AddV2lstm_while_loop_counteradd_9/y:output:0*
_output_shapes
: *
T0�
IdentityIdentity	add_9:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_1Identitylstm_while_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_2Identity	add_8:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_4Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:���������@�

Identity_5Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*'
_output_shapes
:���������@*
T0"F
 split_1_readvariableop_lstm_bias"split_1_readvariableop_lstm_bias_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"�
Ptensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorRtensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0"N
$readvariableop_lstm_recurrent_kernel&readvariableop_lstm_recurrent_kernel_0"!

identity_5Identity_5:output:0"
identityIdentity:output:0".
lstm_strided_slice_1lstm_strided_slice_1_0"F
 split_readvariableop_lstm_kernel"split_readvariableop_lstm_kernel_0*Q
_input_shapes@
>: : : : :���������@:���������@: : :::2 
ReadVariableOpReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp:  : : : : : : : : :	 :
 
�
�
while_body_3802
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0)
%statefulpartitionedcall_lstm_kernel_0'
#statefulpartitionedcall_lstm_bias_03
/statefulpartitionedcall_lstm_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor'
#statefulpartitionedcall_lstm_kernel%
!statefulpartitionedcall_lstm_bias1
-statefulpartitionedcall_lstm_recurrent_kernel��StatefulPartitionedCall�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:����������
StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3%statefulpartitionedcall_lstm_kernel_0#statefulpartitionedcall_lstm_bias_0/statefulpartitionedcall_lstm_recurrent_kernel_0*
Tin

2*M
_output_shapes;
9:���������@:���������@:���������@*+
_gradient_op_typePartitionedCall-3507*L
fGRE
C__inference_lstm_cell_layer_call_and_return_conditional_losses_3407*
Tout
2**
config_proto

GPU 

CPU2J 8�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder StatefulPartitionedCall:output:0*
element_dtype0*
_output_shapes
: G
add/yConst*
value	B :*
dtype0*
_output_shapes
: J
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_1AddV2while_loop_counteradd_1/y:output:0*
_output_shapes
: *
T0Z
IdentityIdentity	add_1:z:0^StatefulPartitionedCall*
T0*
_output_shapes
: k

Identity_1Identitywhile_maximum_iterations^StatefulPartitionedCall*
_output_shapes
: *
T0Z

Identity_2Identityadd:z:0^StatefulPartitionedCall*
_output_shapes
: *
T0�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^StatefulPartitionedCall*
T0*
_output_shapes
: �

Identity_4Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@�

Identity_5Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*'
_output_shapes
:���������@*
T0"!

identity_1Identity_1:output:0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"L
#statefulpartitionedcall_lstm_kernel%statefulpartitionedcall_lstm_kernel_0"$
strided_slice_1strided_slice_1_0"`
-statefulpartitionedcall_lstm_recurrent_kernel/statefulpartitionedcall_lstm_recurrent_kernel_0"H
!statefulpartitionedcall_lstm_bias#statefulpartitionedcall_lstm_bias_0*Q
_input_shapes@
>: : : : :���������@:���������@: : :::22
StatefulPartitionedCallStatefulPartitionedCall:  : : : : : : : : :	 :
 
�
_
A__inference_dropout_layer_call_and_return_conditional_losses_9739

inputs

identity_1N
IdentityIdentityinputs*'
_output_shapes
:���������*
T0[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������"!

identity_1Identity_1:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
�
)__inference_sequential_layer_call_fn_7369

inputs/
+statefulpartitionedcall_features_embeddings'
#statefulpartitionedcall_lstm_kernel%
!statefulpartitionedcall_lstm_bias1
-statefulpartitionedcall_lstm_recurrent_kernel)
%statefulpartitionedcall_lstm_1_kernel'
#statefulpartitionedcall_lstm_1_bias3
/statefulpartitionedcall_lstm_1_recurrent_kernel(
$statefulpartitionedcall_dense_kernel&
"statefulpartitionedcall_dense_bias*
&statefulpartitionedcall_dense_1_kernel(
$statefulpartitionedcall_dense_1_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs+statefulpartitionedcall_features_embeddings#statefulpartitionedcall_lstm_kernel!statefulpartitionedcall_lstm_bias-statefulpartitionedcall_lstm_recurrent_kernel%statefulpartitionedcall_lstm_1_kernel#statefulpartitionedcall_lstm_1_bias/statefulpartitionedcall_lstm_1_recurrent_kernel$statefulpartitionedcall_dense_kernel"statefulpartitionedcall_dense_bias&statefulpartitionedcall_dense_1_kernel$statefulpartitionedcall_dense_1_bias**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:���������*+
_gradient_op_typePartitionedCall-6114*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_6113*
Tout
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*R
_input_shapesA
?:���������(:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : 
�`
�
while_body_8674
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0(
$split_readvariableop_lstm_1_kernel_0(
$split_1_readvariableop_lstm_1_bias_0,
(readvariableop_lstm_1_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor&
"split_readvariableop_lstm_1_kernel&
"split_1_readvariableop_lstm_1_bias*
&readvariableop_lstm_1_recurrent_kernel��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"����@   �
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:���������@G
ConstConst*
_output_shapes
: *
value	B :*
dtype0Q
split/split_dimConst*
_output_shapes
: *
value	B :*
dtype0z
split/ReadVariableOpReadVariableOp$split_readvariableop_lstm_1_kernel_0*
_output_shapes
:	@�*
dtype0�
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
	num_split*<
_output_shapes*
(:@ :@ :@ :@ *
T0~
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:0*'
_output_shapes
:��������� *
T0�
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:1*
T0*'
_output_shapes
:��������� �
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:2*
T0*'
_output_shapes
:��������� �
MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:3*
T0*'
_output_shapes
:��������� I
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
_output_shapes
: *
value	B : *
dtype0x
split_1/ReadVariableOpReadVariableOp$split_1_readvariableop_lstm_1_bias_0*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split*,
_output_shapes
: : : : h
BiasAddBiasAddMatMul:product:0split_1:output:0*'
_output_shapes
:��������� *
T0l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:��������� l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:��������� l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*'
_output_shapes
:��������� *
T0x
ReadVariableOpReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0*
dtype0*
_output_shapes
:	 �d
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
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
Index0*
T0k
MatMul_4MatMulplaceholder_2strided_slice:output:0*
T0*'
_output_shapes
:��������� d
addAddV2BiasAdd:output:0MatMul_4:product:0*
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
: W
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:��������� Y
Add_1AddMul:z:0Const_3:output:0*
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
clip_by_value/yConst*
_output_shapes
: *
valueB
 *    *
dtype0
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*'
_output_shapes
:��������� *
T0�
ReadVariableOp_1ReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0^ReadVariableOp*
dtype0*
_output_shapes
:	 �f
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
strided_slice_2StridedSliceReadVariableOp_1:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:  m
MatMul_5MatMulplaceholder_2strided_slice_2:output:0*
T0*'
_output_shapes
:��������� h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:��������� L
Const_4Const*
_output_shapes
: *
valueB
 *��L>*
dtype0L
Const_5Const*
_output_shapes
: *
valueB
 *   ?*
dtype0[
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:��������� [
Add_3Add	Mul_1:z:0Const_5:output:0*'
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
mul_2Mulclip_by_value_1:z:0placeholder_3*
T0*'
_output_shapes
:��������� �
ReadVariableOp_2ReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0^ReadVariableOp_1*
dtype0*
_output_shapes
:	 �f
strided_slice_3/stackConst*
dtype0*
_output_shapes
:*
valueB"    @   h
strided_slice_3/stack_1Const*
valueB"    `   *
dtype0*
_output_shapes
:h
strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
strided_slice_3StridedSliceReadVariableOp_2:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:  m
MatMul_6MatMulplaceholder_2strided_slice_3:output:0*
T0*'
_output_shapes
:��������� h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:��������� [
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_5AddV2	mul_2:z:0	mul_3:z:0*'
_output_shapes
:��������� *
T0�
ReadVariableOp_3ReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0^ReadVariableOp_2*
dtype0*
_output_shapes
:	 �f
strided_slice_4/stackConst*
valueB"    `   *
dtype0*
_output_shapes
:h
strided_slice_4/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_4/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
strided_slice_4StridedSliceReadVariableOp_3:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
T0*
Index0m
MatMul_7MatMulplaceholder_2strided_slice_4:output:0*
T0*'
_output_shapes
:��������� h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:��������� L
Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_4Mul	add_6:z:0Const_6:output:0*'
_output_shapes
:��������� *
T0[
Add_7Add	Mul_4:z:0Const_7:output:0*'
_output_shapes
:��������� *
T0^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
valueB
 *  �?*
dtype0�
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:��������� V
clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*'
_output_shapes
:��������� *
T0K
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:��������� _
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:��������� �
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_5:z:0*
element_dtype0*
_output_shapes
: I
add_8/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_8AddV2placeholderadd_8/y:output:0*
T0*
_output_shapes
: I
add_9/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_9AddV2while_loop_counteradd_9/y:output:0*
T0*
_output_shapes
: �
IdentityIdentity	add_9:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_1Identitywhile_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_2Identity	add_8:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_4Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*'
_output_shapes
:��������� *
T0�

Identity_5Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:��������� "$
strided_slice_1strided_slice_1_0"J
"split_readvariableop_lstm_1_kernel$split_readvariableop_lstm_1_kernel_0"R
&readvariableop_lstm_1_recurrent_kernel(readvariableop_lstm_1_recurrent_kernel_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"J
"split_1_readvariableop_lstm_1_bias$split_1_readvariableop_lstm_1_bias_0"
identityIdentity:output:0"!

identity_5Identity_5:output:0*Q
_input_shapes@
>: : : : :��������� :��������� : : :::20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp2 
ReadVariableOpReadVariableOp: : : : : :	 :
 :  : : : 
�
�
?__inference_dense_layer_call_and_return_conditional_losses_5969

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
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:��������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�`
�
while_body_7526
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0&
"split_readvariableop_lstm_kernel_0&
"split_1_readvariableop_lstm_bias_0*
&readvariableop_lstm_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor$
 split_readvariableop_lstm_kernel$
 split_1_readvariableop_lstm_bias(
$readvariableop_lstm_recurrent_kernel��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:���������G
ConstConst*
dtype0*
_output_shapes
: *
value	B :Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: x
split/ReadVariableOpReadVariableOp"split_readvariableop_lstm_kernel_0*
dtype0*
_output_shapes
:	��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split*<
_output_shapes*
(:@:@:@:@~
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:0*'
_output_shapes
:���������@*
T0�
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:1*'
_output_shapes
:���������@*
T0�
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:2*'
_output_shapes
:���������@*
T0�
MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:3*
T0*'
_output_shapes
:���������@I
Const_1Const*
_output_shapes
: *
value	B :*
dtype0S
split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: v
split_1/ReadVariableOpReadVariableOp"split_1_readvariableop_lstm_bias_0*
_output_shapes	
:�*
dtype0�
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
	num_split*,
_output_shapes
:@:@:@:@*
T0h
BiasAddBiasAddMatMul:product:0split_1:output:0*'
_output_shapes
:���������@*
T0l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:���������@l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:���������@l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:���������@v
ReadVariableOpReadVariableOp&readvariableop_lstm_recurrent_kernel_0*
_output_shapes
:	@�*
dtype0d
strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB"    @   f
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:@@k
MatMul_4MatMulplaceholder_2strided_slice:output:0*'
_output_shapes
:���������@*
T0d
addAddV2BiasAdd:output:0MatMul_4:product:0*'
_output_shapes
:���������@*
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
: W
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:���������@Y
Add_1AddMul:z:0Const_3:output:0*'
_output_shapes
:���������@*
T0\
clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*'
_output_shapes
:���������@*
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
:���������@�
ReadVariableOp_1ReadVariableOp&readvariableop_lstm_recurrent_kernel_0^ReadVariableOp*
dtype0*
_output_shapes
:	@�f
strided_slice_2/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB"    �   h
strided_slice_2/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_slice_2StridedSliceReadVariableOp_1:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:@@*
Index0*
T0m
MatMul_5MatMulplaceholder_2strided_slice_2:output:0*
T0*'
_output_shapes
:���������@h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:���������@L
Const_4Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_4:output:0*'
_output_shapes
:���������@*
T0[
Add_3Add	Mul_1:z:0Const_5:output:0*'
_output_shapes
:���������@*
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
:���������@V
clip_by_value_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:���������@b
mul_2Mulclip_by_value_1:z:0placeholder_3*
T0*'
_output_shapes
:���������@�
ReadVariableOp_2ReadVariableOp&readvariableop_lstm_recurrent_kernel_0^ReadVariableOp_1*
dtype0*
_output_shapes
:	@�f
strided_slice_3/stackConst*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_3/stack_1Const*
valueB"    �   *
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
end_mask*
_output_shapes

:@@m
MatMul_6MatMulplaceholder_2strided_slice_3:output:0*
T0*'
_output_shapes
:���������@h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*'
_output_shapes
:���������@*
T0I
TanhTanh	add_4:z:0*'
_output_shapes
:���������@*
T0[
mul_3Mulclip_by_value:z:0Tanh:y:0*'
_output_shapes
:���������@*
T0V
add_5AddV2	mul_2:z:0	mul_3:z:0*'
_output_shapes
:���������@*
T0�
ReadVariableOp_3ReadVariableOp&readvariableop_lstm_recurrent_kernel_0^ReadVariableOp_2*
dtype0*
_output_shapes
:	@�f
strided_slice_4/stackConst*
dtype0*
_output_shapes
:*
valueB"    �   h
strided_slice_4/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_4StridedSliceReadVariableOp_3:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:@@m
MatMul_7MatMulplaceholder_2strided_slice_4:output:0*
T0*'
_output_shapes
:���������@h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:���������@L
Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_7Const*
_output_shapes
: *
valueB
 *   ?*
dtype0[
Mul_4Mul	add_6:z:0Const_6:output:0*'
_output_shapes
:���������@*
T0[
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:���������@^
clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:���������@V
clip_by_value_2/yConst*
dtype0*
_output_shapes
: *
valueB
 *    �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:���������@K
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:���������@_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*'
_output_shapes
:���������@*
T0�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_5:z:0*
element_dtype0*
_output_shapes
: I
add_8/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_8AddV2placeholderadd_8/y:output:0*
_output_shapes
: *
T0I
add_9/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_9AddV2while_loop_counteradd_9/y:output:0*
T0*
_output_shapes
: �
IdentityIdentity	add_9:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_1Identitywhile_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_2Identity	add_8:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_4Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:���������@�

Identity_5Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:���������@"F
 split_readvariableop_lstm_kernel"split_readvariableop_lstm_kernel_0"$
strided_slice_1strided_slice_1_0"F
 split_1_readvariableop_lstm_bias"split_1_readvariableop_lstm_bias_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"N
$readvariableop_lstm_recurrent_kernel&readvariableop_lstm_recurrent_kernel_0"
identityIdentity:output:0"!

identity_5Identity_5:output:0*Q
_input_shapes@
>: : : : :���������@:���������@: : :::2 
ReadVariableOpReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp: :	 :
 :  : : : : : : : 
�
`
A__inference_dropout_layer_call_and_return_conditional_losses_9734

inputs
identity�Q
dropout/rateConst*
valueB
 *   ?*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*'
_output_shapes
:���������*
T0*
dtype0�
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:����������
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:���������R
dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: �
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*'
_output_shapes
:���������a
dropout/mulMulinputsdropout/truediv:z:0*
T0*'
_output_shapes
:���������o
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:���������i
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������Y
IdentityIdentitydropout/mul_1:z:0*'
_output_shapes
:���������*
T0"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
�
+__inference_lstm_cell_1_layer_call_fn_10187

inputs
states_0
states_1)
%statefulpartitionedcall_lstm_1_kernel'
#statefulpartitionedcall_lstm_1_bias3
/statefulpartitionedcall_lstm_1_recurrent_kernel
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1%statefulpartitionedcall_lstm_1_kernel#statefulpartitionedcall_lstm_1_bias/statefulpartitionedcall_lstm_1_recurrent_kernel*N
fIRG
E__inference_lstm_cell_1_layer_call_and_return_conditional_losses_4247*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin

2*M
_output_shapes;
9:��������� :��������� :��������� *+
_gradient_op_typePartitionedCall-4271�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� �

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*'
_output_shapes
:��������� *
T0�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:���������@:��������� :��������� :::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0:($
"
_user_specified_name
states/1: : : 
��
�
>__inference_lstm_layer_call_and_return_conditional_losses_5076

inputs$
 split_readvariableop_lstm_kernel$
 split_1_readvariableop_lstm_bias(
$readvariableop_lstm_recurrent_kernel
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0_
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
value	B :@_
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
value	B :@*
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
:���������@*
T0O
zeros_1/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
dtype0*
_output_shapes
: *
value
B :�_
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
_output_shapes
: *
T0R
zeros_1/packed/1Const*
dtype0*
_output_shapes
: *
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*'
_output_shapes
:���������@*
T0c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*+
_output_shapes
:(���������*
T0D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
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
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*'
_output_shapes
:���������G
ConstConst*
_output_shapes
: *
value	B :*
dtype0Q
split/split_dimConst*
_output_shapes
: *
value	B :*
dtype0v
split/ReadVariableOpReadVariableOp split_readvariableop_lstm_kernel*
dtype0*
_output_shapes
:	��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
	num_split*<
_output_shapes*
(:@:@:@:@*
T0l
MatMulMatMulstrided_slice_2:output:0split:output:0*
T0*'
_output_shapes
:���������@n
MatMul_1MatMulstrided_slice_2:output:0split:output:1*
T0*'
_output_shapes
:���������@n
MatMul_2MatMulstrided_slice_2:output:0split:output:2*'
_output_shapes
:���������@*
T0n
MatMul_3MatMulstrided_slice_2:output:0split:output:3*
T0*'
_output_shapes
:���������@I
Const_1Const*
dtype0*
_output_shapes
: *
value	B :S
split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: t
split_1/ReadVariableOpReadVariableOp split_1_readvariableop_lstm_bias*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
	num_split*,
_output_shapes
:@:@:@:@*
T0h
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:���������@l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:���������@l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*'
_output_shapes
:���������@*
T0l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*'
_output_shapes
:���������@*
T0t
ReadVariableOpReadVariableOp$readvariableop_lstm_recurrent_kernel*
dtype0*
_output_shapes
:	@�f
strided_slice_3/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_3/stack_1Const*
valueB"    @   *
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

:@@n
MatMul_4MatMulzeros:output:0strided_slice_3:output:0*
T0*'
_output_shapes
:���������@d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:���������@L
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
: W
MulMuladd:z:0Const_2:output:0*'
_output_shapes
:���������@*
T0Y
Add_1AddMul:z:0Const_3:output:0*
T0*'
_output_shapes
:���������@\
clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������@T
clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������@�
ReadVariableOp_1ReadVariableOp$readvariableop_lstm_recurrent_kernel^ReadVariableOp*
dtype0*
_output_shapes
:	@�f
strided_slice_4/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_4/stack_1Const*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_4StridedSliceReadVariableOp_1:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
end_mask*
_output_shapes

:@@*
Index0*
T0*

begin_maskn
MatMul_5MatMulzeros:output:0strided_slice_4:output:0*
T0*'
_output_shapes
:���������@h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:���������@L
Const_4Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_5Const*
_output_shapes
: *
valueB
 *   ?*
dtype0[
Mul_1Mul	add_2:z:0Const_4:output:0*'
_output_shapes
:���������@*
T0[
Add_3Add	Mul_1:z:0Const_5:output:0*'
_output_shapes
:���������@*
T0^
clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*'
_output_shapes
:���������@*
T0V
clip_by_value_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*'
_output_shapes
:���������@*
T0e
mul_2Mulclip_by_value_1:z:0zeros_1:output:0*'
_output_shapes
:���������@*
T0�
ReadVariableOp_2ReadVariableOp$readvariableop_lstm_recurrent_kernel^ReadVariableOp_1*
dtype0*
_output_shapes
:	@�f
strided_slice_5/stackConst*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_5/stack_1Const*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_5/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_5StridedSliceReadVariableOp_2:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
end_mask*
_output_shapes

:@@*
T0*
Index0*

begin_maskn
MatMul_6MatMulzeros:output:0strided_slice_5:output:0*'
_output_shapes
:���������@*
T0h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*'
_output_shapes
:���������@*
T0I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:���������@[
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:���������@V
add_5AddV2	mul_2:z:0	mul_3:z:0*'
_output_shapes
:���������@*
T0�
ReadVariableOp_3ReadVariableOp$readvariableop_lstm_recurrent_kernel^ReadVariableOp_2*
dtype0*
_output_shapes
:	@�f
strided_slice_6/stackConst*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_6/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_6/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_6StridedSliceReadVariableOp_3:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:@@*
T0*
Index0n
MatMul_7MatMulzeros:output:0strided_slice_6:output:0*
T0*'
_output_shapes
:���������@h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*'
_output_shapes
:���������@*
T0L
Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_7Const*
dtype0*
_output_shapes
: *
valueB
 *   ?[
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:���������@[
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:���������@^
clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:���������@V
clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:���������@K
Tanh_1Tanh	add_5:z:0*'
_output_shapes
:���������@*
T0_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*'
_output_shapes
:���������@*
T0n
TensorArrayV2_1/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"����@   �
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 split_readvariableop_lstm_kernel split_1_readvariableop_lstm_bias$readvariableop_lstm_recurrent_kernel^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_4922*
_num_original_outputs*
bodyR
while_body_4923*L
_output_shapes:
8: : : : :���������@:���������@: : : : : *K
output_shapes:
8: : : : :���������@:���������@: : : : : *
T
2K
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
:���������@^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:���������@M
while/Identity_6Identitywhile:output:6*
_output_shapes
: *
T0M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
valueB"����@   *
dtype0�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:(���������@h
strided_slice_7/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:a
strided_slice_7/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_7StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:���������@e
transpose_1/permConst*
dtype0*
_output_shapes
:*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*+
_output_shapes
:���������(@*
T0�
IdentityIdentitytranspose_1:y:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp^while*+
_output_shapes
:���������(@*
T0"
identityIdentity:output:0*6
_input_shapes%
#:���������(:::2$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp2
whilewhile2 
ReadVariableOpReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_2:& "
 
_user_specified_nameinputs: : : 
�
�
while_cond_3801
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
lstm_kernel
	lstm_bias
lstm_recurrent_kernel
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
identityIdentity:output:0*Q
_input_shapes@
>: : : : :���������@:���������@: : :::: : : : : :	 :
 :  : : : 
�L
�
C__inference_lstm_cell_layer_call_and_return_conditional_losses_9858

inputs
states_0
states_1$
 split_readvariableop_lstm_kernel$
 split_1_readvariableop_lstm_bias(
$readvariableop_lstm_recurrent_kernel
identity

identity_1

identity_2��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOpG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: v
split/ReadVariableOpReadVariableOp split_readvariableop_lstm_kernel*
dtype0*
_output_shapes
:	��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
	num_split*<
_output_shapes*
(:@:@:@:@*
T0Z
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:���������@\
MatMul_1MatMulinputssplit:output:1*'
_output_shapes
:���������@*
T0\
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:���������@\
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:���������@I
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: t
split_1/ReadVariableOpReadVariableOp split_1_readvariableop_lstm_bias*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
	num_split*,
_output_shapes
:@:@:@:@*
T0h
BiasAddBiasAddMatMul:product:0split_1:output:0*'
_output_shapes
:���������@*
T0l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:���������@l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:���������@l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:���������@t
ReadVariableOpReadVariableOp$readvariableop_lstm_recurrent_kernel*
dtype0*
_output_shapes
:	@�d
strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:f
strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:@@f
MatMul_4MatMulstates_0strided_slice:output:0*
T0*'
_output_shapes
:���������@d
addAddV2BiasAdd:output:0MatMul_4:product:0*'
_output_shapes
:���������@*
T0L
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
dtype0W
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:���������@Y
Add_1AddMul:z:0Const_3:output:0*
T0*'
_output_shapes
:���������@\
clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*'
_output_shapes
:���������@*
T0T
clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*'
_output_shapes
:���������@*
T0�
ReadVariableOp_1ReadVariableOp$readvariableop_lstm_recurrent_kernel^ReadVariableOp*
dtype0*
_output_shapes
:	@�f
strided_slice_1/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_1/stack_1Const*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:@@h
MatMul_5MatMulstates_0strided_slice_1:output:0*
T0*'
_output_shapes
:���������@h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:���������@L
Const_4Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:���������@[
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*'
_output_shapes
:���������@^
clip_by_value_1/Minimum/yConst*
dtype0*
_output_shapes
: *
valueB
 *  �?�
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:���������@V
clip_by_value_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:���������@]
mul_2Mulclip_by_value_1:z:0states_1*
T0*'
_output_shapes
:���������@�
ReadVariableOp_2ReadVariableOp$readvariableop_lstm_recurrent_kernel^ReadVariableOp_1*
dtype0*
_output_shapes
:	@�f
strided_slice_2/stackConst*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_1Const*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
end_mask*
_output_shapes

:@@*
Index0*
T0*

begin_maskh
MatMul_6MatMulstates_0strided_slice_2:output:0*'
_output_shapes
:���������@*
T0h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*'
_output_shapes
:���������@*
T0I
TanhTanh	add_4:z:0*'
_output_shapes
:���������@*
T0[
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:���������@V
add_5AddV2	mul_2:z:0	mul_3:z:0*'
_output_shapes
:���������@*
T0�
ReadVariableOp_3ReadVariableOp$readvariableop_lstm_recurrent_kernel^ReadVariableOp_2*
dtype0*
_output_shapes
:	@�f
strided_slice_3/stackConst*
valueB"    �   *
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
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
end_mask*
_output_shapes

:@@*
Index0*
T0*

begin_maskh
MatMul_7MatMulstates_0strided_slice_3:output:0*
T0*'
_output_shapes
:���������@h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:���������@L
Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:���������@[
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:���������@^
clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*'
_output_shapes
:���������@*
T0V
clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*'
_output_shapes
:���������@*
T0K
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:���������@_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:���������@�
IdentityIdentity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*'
_output_shapes
:���������@*
T0�

Identity_1Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*'
_output_shapes
:���������@*
T0�

Identity_2Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*'
_output_shapes
:���������@*
T0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:���������:���������@:���������@:::2 
ReadVariableOpReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0:($
"
_user_specified_name
states/1: : : 
�`
�
while_body_5503
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0(
$split_readvariableop_lstm_1_kernel_0(
$split_1_readvariableop_lstm_1_bias_0,
(readvariableop_lstm_1_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor&
"split_readvariableop_lstm_1_kernel&
"split_1_readvariableop_lstm_1_bias*
&readvariableop_lstm_1_recurrent_kernel��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����@   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:���������@G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: z
split/ReadVariableOpReadVariableOp$split_readvariableop_lstm_1_kernel_0*
dtype0*
_output_shapes
:	@��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split*<
_output_shapes*
(:@ :@ :@ :@ ~
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:0*'
_output_shapes
:��������� *
T0�
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:1*'
_output_shapes
:��������� *
T0�
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:2*'
_output_shapes
:��������� *
T0�
MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:3*
T0*'
_output_shapes
:��������� I
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: x
split_1/ReadVariableOpReadVariableOp$split_1_readvariableop_lstm_1_bias_0*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
	num_split*,
_output_shapes
: : : : *
T0h
BiasAddBiasAddMatMul:product:0split_1:output:0*'
_output_shapes
:��������� *
T0l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:��������� l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*'
_output_shapes
:��������� *
T0l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:��������� x
ReadVariableOpReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0*
dtype0*
_output_shapes
:	 �d
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"        f
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
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:  k
MatMul_4MatMulplaceholder_2strided_slice:output:0*
T0*'
_output_shapes
:��������� d
addAddV2BiasAdd:output:0MatMul_4:product:0*
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
: W
MulMuladd:z:0Const_2:output:0*'
_output_shapes
:��������� *
T0Y
Add_1AddMul:z:0Const_3:output:0*'
_output_shapes
:��������� *
T0\
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
clip_by_value/yConst*
_output_shapes
: *
valueB
 *    *
dtype0
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_1ReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0^ReadVariableOp*
dtype0*
_output_shapes
:	 �f
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
dtype0*
_output_shapes
:*
valueB"      �
strided_slice_2StridedSliceReadVariableOp_1:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
end_mask*
_output_shapes

:  *
T0*
Index0*

begin_maskm
MatMul_5MatMulplaceholder_2strided_slice_2:output:0*'
_output_shapes
:��������� *
T0h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*'
_output_shapes
:��������� *
T0L
Const_4Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_5Const*
dtype0*
_output_shapes
: *
valueB
 *   ?[
Mul_1Mul	add_2:z:0Const_4:output:0*'
_output_shapes
:��������� *
T0[
Add_3Add	Mul_1:z:0Const_5:output:0*'
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
mul_2Mulclip_by_value_1:z:0placeholder_3*'
_output_shapes
:��������� *
T0�
ReadVariableOp_2ReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0^ReadVariableOp_1*
dtype0*
_output_shapes
:	 �f
strided_slice_3/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_3/stack_1Const*
valueB"    `   *
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
end_mask*
_output_shapes

:  *
T0*
Index0m
MatMul_6MatMulplaceholder_2strided_slice_3:output:0*
T0*'
_output_shapes
:��������� h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_4:z:0*'
_output_shapes
:��������� *
T0[
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_3ReadVariableOp(readvariableop_lstm_1_recurrent_kernel_0^ReadVariableOp_2*
dtype0*
_output_shapes
:	 �f
strided_slice_4/stackConst*
valueB"    `   *
dtype0*
_output_shapes
:h
strided_slice_4/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_4StridedSliceReadVariableOp_3:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
_output_shapes

:  *
Index0*
T0*

begin_mask*
end_maskm
MatMul_7MatMulplaceholder_2strided_slice_4:output:0*'
_output_shapes
:��������� *
T0h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:��������� L
Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_4Mul	add_6:z:0Const_6:output:0*'
_output_shapes
:��������� *
T0[
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:��������� ^
clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*'
_output_shapes
:��������� *
T0V
clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*'
_output_shapes
:��������� *
T0K
Tanh_1Tanh	add_5:z:0*'
_output_shapes
:��������� *
T0_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:��������� �
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_5:z:0*
element_dtype0*
_output_shapes
: I
add_8/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_8AddV2placeholderadd_8/y:output:0*
T0*
_output_shapes
: I
add_9/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_9AddV2while_loop_counteradd_9/y:output:0*
T0*
_output_shapes
: �
IdentityIdentity	add_9:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_1Identitywhile_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_2Identity	add_8:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_4Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*'
_output_shapes
:��������� *
T0�

Identity_5Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*'
_output_shapes
:��������� *
T0"!

identity_4Identity_4:output:0"J
"split_1_readvariableop_lstm_1_bias$split_1_readvariableop_lstm_1_bias_0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"$
strided_slice_1strided_slice_1_0"J
"split_readvariableop_lstm_1_kernel$split_readvariableop_lstm_1_kernel_0"R
&readvariableop_lstm_1_recurrent_kernel(readvariableop_lstm_1_recurrent_kernel_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*Q
_input_shapes@
>: : : : :��������� :��������� : : :::2 
ReadVariableOpReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp: : : : :	 :
 :  : : : : 
��
�
@__inference_lstm_1_layer_call_and_return_conditional_losses_9401

inputs&
"split_readvariableop_lstm_1_kernel&
"split_1_readvariableop_lstm_1_bias*
&readvariableop_lstm_1_recurrent_kernel
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOp�while;
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
: *
Index0*
T0*
shrink_axis_maskM
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
:��������� O
zeros_1/mul/yConst*
value	B : *
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
_output_shapes
: *
T0Q
zeros_1/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
_output_shapes
: *
value	B : *
dtype0w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
_output_shapes
:*
T0R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
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
:(���������@*
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
_output_shapes
: *
T0*
Index0*
shrink_axis_maskf
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
valueB"����@   *
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
strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:���������@*
Index0*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: x
split/ReadVariableOpReadVariableOp"split_readvariableop_lstm_1_kernel*
dtype0*
_output_shapes
:	@��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split*<
_output_shapes*
(:@ :@ :@ :@ l
MatMulMatMulstrided_slice_2:output:0split:output:0*'
_output_shapes
:��������� *
T0n
MatMul_1MatMulstrided_slice_2:output:0split:output:1*'
_output_shapes
:��������� *
T0n
MatMul_2MatMulstrided_slice_2:output:0split:output:2*'
_output_shapes
:��������� *
T0n
MatMul_3MatMulstrided_slice_2:output:0split:output:3*
T0*'
_output_shapes
:��������� I
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: v
split_1/ReadVariableOpReadVariableOp"split_1_readvariableop_lstm_1_bias*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
	num_split*,
_output_shapes
: : : : *
T0h
BiasAddBiasAddMatMul:product:0split_1:output:0*'
_output_shapes
:��������� *
T0l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*'
_output_shapes
:��������� *
T0l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:��������� l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*'
_output_shapes
:��������� *
T0v
ReadVariableOpReadVariableOp&readvariableop_lstm_1_recurrent_kernel*
dtype0*
_output_shapes
:	 �f
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

:  *
T0*
Index0n
MatMul_4MatMulzeros:output:0strided_slice_3:output:0*
T0*'
_output_shapes
:��������� d
addAddV2BiasAdd:output:0MatMul_4:product:0*'
_output_shapes
:��������� *
T0L
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
 *   ?W
MulMuladd:z:0Const_2:output:0*'
_output_shapes
:��������� *
T0Y
Add_1AddMul:z:0Const_3:output:0*'
_output_shapes
:��������� *
T0\
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
:��������� �
ReadVariableOp_1ReadVariableOp&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp*
dtype0*
_output_shapes
:	 �f
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
strided_slice_4StridedSliceReadVariableOp_1:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:  n
MatMul_5MatMulzeros:output:0strided_slice_4:output:0*
T0*'
_output_shapes
:��������� h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:��������� L
Const_4Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:��������� [
Add_3Add	Mul_1:z:0Const_5:output:0*
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
:��������� e
mul_2Mulclip_by_value_1:z:0zeros_1:output:0*'
_output_shapes
:��������� *
T0�
ReadVariableOp_2ReadVariableOp&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp_1*
dtype0*
_output_shapes
:	 �f
strided_slice_5/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_5/stack_1Const*
valueB"    `   *
dtype0*
_output_shapes
:h
strided_slice_5/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_5StridedSliceReadVariableOp_2:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
_output_shapes

:  *
Index0*
T0*

begin_mask*
end_maskn
MatMul_6MatMulzeros:output:0strided_slice_5:output:0*
T0*'
_output_shapes
:��������� h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*'
_output_shapes
:��������� *
T0I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:��������� [
mul_3Mulclip_by_value:z:0Tanh:y:0*'
_output_shapes
:��������� *
T0V
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_3ReadVariableOp&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp_2*
dtype0*
_output_shapes
:	 �f
strided_slice_6/stackConst*
valueB"    `   *
dtype0*
_output_shapes
:h
strided_slice_6/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_6/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_6StridedSliceReadVariableOp_3:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:  n
MatMul_7MatMulzeros:output:0strided_slice_6:output:0*
T0*'
_output_shapes
:��������� h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:��������� L
Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:��������� [
Add_7Add	Mul_4:z:0Const_7:output:0*'
_output_shapes
:��������� *
T0^
clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*'
_output_shapes
:��������� *
T0V
clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:��������� K
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:��������� _
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
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
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"split_readvariableop_lstm_1_kernel"split_1_readvariableop_lstm_1_bias&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *
T
2*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_9247*
_num_original_outputs*
bodyR
while_body_9248K
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
while/Identity_4Identitywhile:output:4*'
_output_shapes
:��������� *
T0^
while/Identity_5Identitywhile:output:5*'
_output_shapes
:��������� *
T0M
while/Identity_6Identitywhile:output:6*
_output_shapes
: *
T0M
while/Identity_7Identitywhile:output:7*
_output_shapes
: *
T0M
while/Identity_8Identitywhile:output:8*
_output_shapes
: *
T0M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
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
:(��������� h
strided_slice_7/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:a
strided_slice_7/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_7StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*'
_output_shapes
:��������� *
T0*
Index0*
shrink_axis_maske
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������( �
IdentityIdentitystrided_slice_7:output:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp^while*'
_output_shapes
:��������� *
T0"
identityIdentity:output:0*6
_input_shapes%
#:���������(@:::20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp2
whilewhile2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs: : : 
�	
�
A__inference_dense_1_layer_call_and_return_conditional_losses_9760

inputs(
$matmul_readvariableop_dense_1_kernel'
#biasadd_readvariableop_dense_1_bias
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpz
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_1_kernel*
dtype0*
_output_shapes

:i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_1_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*'
_output_shapes
:���������*
T0�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
�
while_cond_8952
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
lstm_1_kernel
lstm_1_bias
lstm_1_recurrent_kernel
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
identityIdentity:output:0*Q
_input_shapes@
>: : : : :��������� :��������� : : :::: : :	 :
 :  : : : : : : 
�
�
%__inference_lstm_1_layer_call_fn_9114
inputs_0)
%statefulpartitionedcall_lstm_1_kernel'
#statefulpartitionedcall_lstm_1_bias3
/statefulpartitionedcall_lstm_1_recurrent_kernel
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0%statefulpartitionedcall_lstm_1_kernel#statefulpartitionedcall_lstm_1_bias/statefulpartitionedcall_lstm_1_recurrent_kernel*'
_output_shapes
:��������� *
Tin
2*+
_gradient_op_typePartitionedCall-4625*I
fDRB
@__inference_lstm_1_layer_call_and_return_conditional_losses_4624*
Tout
2**
config_proto

GPU 

CPU2J 8�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:��������� *
T0"
identityIdentity:output:0*?
_input_shapes.
,:������������������@:::22
StatefulPartitionedCallStatefulPartitionedCall: : :( $
"
_user_specified_name
inputs/0: 
�
`
A__inference_dropout_layer_call_and_return_conditional_losses_6008

inputs
identity�Q
dropout/rateConst*
_output_shapes
: *
valueB
 *   ?*
dtype0C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
_output_shapes
: *
valueB
 *  �?*
dtype0�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:����������
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:����������
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*'
_output_shapes
:���������*
T0R
dropout/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0�
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*'
_output_shapes
:���������a
dropout/mulMulinputsdropout/truediv:z:0*
T0*'
_output_shapes
:���������o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*'
_output_shapes
:���������*

SrcT0
i
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������Y
IdentityIdentitydropout/mul_1:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
�
lstm_while_cond_6909
lstm_while_loop_counter!
lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_lstm_strided_slice_10
,lstm_tensorarrayunstack_tensorlistfromtensor
lstm_kernel
	lstm_bias
lstm_recurrent_kernel
identity
U
LessLessplaceholderless_lstm_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :���������@:���������@: : :::: : : : : : : :	 :
 :  : 
Ă
�
@__inference_lstm_1_layer_call_and_return_conditional_losses_9106
inputs_0&
"split_readvariableop_lstm_1_kernel&
"split_1_readvariableop_lstm_1_bias*
&readvariableop_lstm_1_recurrent_kernel
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOp�while=
ShapeShapeinputs_0*
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
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0M
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
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
_output_shapes
:*
T0*
NP
zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*'
_output_shapes
:��������� *
T0O
zeros_1/mul/yConst*
value	B : *
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B : *
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������@D
Shape_1Shapetranspose:y:0*
_output_shapes
:*
T0_
strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB: a
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
valueB"����@   *
dtype0*
_output_shapes
:�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB: a
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
:���������@*
T0*
Index0G
ConstConst*
dtype0*
_output_shapes
: *
value	B :Q
split/split_dimConst*
_output_shapes
: *
value	B :*
dtype0x
split/ReadVariableOpReadVariableOp"split_readvariableop_lstm_1_kernel*
_output_shapes
:	@�*
dtype0�
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split*<
_output_shapes*
(:@ :@ :@ :@ l
MatMulMatMulstrided_slice_2:output:0split:output:0*
T0*'
_output_shapes
:��������� n
MatMul_1MatMulstrided_slice_2:output:0split:output:1*'
_output_shapes
:��������� *
T0n
MatMul_2MatMulstrided_slice_2:output:0split:output:2*'
_output_shapes
:��������� *
T0n
MatMul_3MatMulstrided_slice_2:output:0split:output:3*
T0*'
_output_shapes
:��������� I
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: v
split_1/ReadVariableOpReadVariableOp"split_1_readvariableop_lstm_1_bias*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split*,
_output_shapes
: : : : h
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:��������� l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:��������� l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*'
_output_shapes
:��������� *
T0l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:��������� v
ReadVariableOpReadVariableOp&readvariableop_lstm_1_recurrent_kernel*
dtype0*
_output_shapes
:	 �f
strided_slice_3/stackConst*
_output_shapes
:*
valueB"        *
dtype0h
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

:  n
MatMul_4MatMulzeros:output:0strided_slice_3:output:0*
T0*'
_output_shapes
:��������� d
addAddV2BiasAdd:output:0MatMul_4:product:0*
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
: W
MulMuladd:z:0Const_2:output:0*'
_output_shapes
:��������� *
T0Y
Add_1AddMul:z:0Const_3:output:0*'
_output_shapes
:��������� *
T0\
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
:��������� �
ReadVariableOp_1ReadVariableOp&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp*
dtype0*
_output_shapes
:	 �f
strided_slice_4/stackConst*
_output_shapes
:*
valueB"        *
dtype0h
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

:  *
T0*
Index0n
MatMul_5MatMulzeros:output:0strided_slice_4:output:0*'
_output_shapes
:��������� *
T0h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:��������� L
Const_4Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_4:output:0*'
_output_shapes
:��������� *
T0[
Add_3Add	Mul_1:z:0Const_5:output:0*
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
:��������� e
mul_2Mulclip_by_value_1:z:0zeros_1:output:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_2ReadVariableOp&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp_1*
dtype0*
_output_shapes
:	 �f
strided_slice_5/stackConst*
dtype0*
_output_shapes
:*
valueB"    @   h
strided_slice_5/stack_1Const*
valueB"    `   *
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
end_mask*
_output_shapes

:  *
Index0*
T0n
MatMul_6MatMulzeros:output:0strided_slice_5:output:0*'
_output_shapes
:��������� *
T0h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:��������� [
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_3ReadVariableOp&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp_2*
dtype0*
_output_shapes
:	 �f
strided_slice_6/stackConst*
valueB"    `   *
dtype0*
_output_shapes
:h
strided_slice_6/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_6/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_6StridedSliceReadVariableOp_3:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
Index0*
T0n
MatMul_7MatMulzeros:output:0strided_slice_6:output:0*'
_output_shapes
:��������� *
T0h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*'
_output_shapes
:��������� *
T0L
Const_6Const*
dtype0*
_output_shapes
: *
valueB
 *��L>L
Const_7Const*
dtype0*
_output_shapes
: *
valueB
 *   ?[
Mul_4Mul	add_6:z:0Const_6:output:0*'
_output_shapes
:��������� *
T0[
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:��������� ^
clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:��������� V
clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*'
_output_shapes
:��������� *
T0K
Tanh_1Tanh	add_5:z:0*'
_output_shapes
:��������� *
T0_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
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
while/maximum_iterationsConst*
dtype0*
_output_shapes
: *
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
value	B : *
dtype0�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"split_readvariableop_lstm_1_kernel"split_1_readvariableop_lstm_1_bias&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_8952*
_num_original_outputs*
bodyR
while_body_8953*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *
T
2*K
output_shapes:
8: : : : :��������� :��������� : : : : : K
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
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*'
_output_shapes
:��������� *
T0^
while/Identity_5Identitywhile:output:5*'
_output_shapes
:��������� *
T0M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
_output_shapes
: *
T0M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
_output_shapes
: *
T0�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :������������������ h
strided_slice_7/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:a
strided_slice_7/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_7StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
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
 :������������������ �
IdentityIdentitystrided_slice_7:output:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp^while*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*?
_input_shapes.
,:������������������@:::2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp2
whilewhile2 
ReadVariableOpReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:( $
"
_user_specified_name
inputs/0: : : 
�
�
!sequential_lstm_1_while_cond_3104(
$sequential_lstm_1_while_loop_counter.
*sequential_lstm_1_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3*
&less_sequential_lstm_1_strided_slice_1=
9sequential_lstm_1_tensorarrayunstack_tensorlistfromtensor
lstm_1_kernel
lstm_1_bias
lstm_1_recurrent_kernel
identity
b
LessLessplaceholder&less_sequential_lstm_1_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :��������� :��������� : : ::::	 :
 :  : : : : : : : : 
�
�
#__inference_lstm_layer_call_fn_7974
inputs_0'
#statefulpartitionedcall_lstm_kernel%
!statefulpartitionedcall_lstm_bias1
-statefulpartitionedcall_lstm_recurrent_kernel
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0#statefulpartitionedcall_lstm_kernel!statefulpartitionedcall_lstm_bias-statefulpartitionedcall_lstm_recurrent_kernel*+
_gradient_op_typePartitionedCall-4016*G
fBR@
>__inference_lstm_layer_call_and_return_conditional_losses_4015*
Tout
2**
config_proto

GPU 

CPU2J 8*4
_output_shapes"
 :������������������@*
Tin
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :������������������@"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0: : : 
�
�
while_cond_4922
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
lstm_kernel
	lstm_bias
lstm_recurrent_kernel
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
identityIdentity:output:0*Q
_input_shapes@
>: : : : :���������@:���������@: : :::: : : : : : :	 :
 :  : : 
�
�
#__inference_lstm_layer_call_fn_7966
inputs_0'
#statefulpartitionedcall_lstm_kernel%
!statefulpartitionedcall_lstm_bias1
-statefulpartitionedcall_lstm_recurrent_kernel
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0#statefulpartitionedcall_lstm_kernel!statefulpartitionedcall_lstm_bias-statefulpartitionedcall_lstm_recurrent_kernel**
config_proto

GPU 

CPU2J 8*
Tin
2*4
_output_shapes"
 :������������������@*+
_gradient_op_typePartitionedCall-3879*G
fBR@
>__inference_lstm_layer_call_and_return_conditional_losses_3878*
Tout
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :������������������@"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0: : : 
�
B
&__inference_dropout_layer_call_fn_9749

inputs
identity�
PartitionedCallPartitionedCallinputs*+
_gradient_op_typePartitionedCall-6029*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_6016*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:���������`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
�
lstm_1_while_cond_7184
lstm_1_while_loop_counter#
lstm_1_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_lstm_1_strided_slice_12
.lstm_1_tensorarrayunstack_tensorlistfromtensor
lstm_1_kernel
lstm_1_bias
lstm_1_recurrent_kernel
identity
W
LessLessplaceholderless_lstm_1_strided_slice_1*
_output_shapes
: *
T0?
IdentityIdentityLess:z:0*
_output_shapes
: *
T0
"
identityIdentity:output:0*Q
_input_shapes@
>: : : : :��������� :��������� : : :::: : : : :	 :
 :  : : : : 
��
�
@__inference_lstm_1_layer_call_and_return_conditional_losses_5935

inputs&
"split_readvariableop_lstm_1_kernel&
"split_1_readvariableop_lstm_1_bias*
&readvariableop_lstm_1_recurrent_kernel
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOp�while;
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
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_maskM
zeros/mul/yConst*
dtype0*
_output_shapes
: *
value	B : _
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
zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0l
zerosFillzeros/packed:output:0zeros/Const:output:0*'
_output_shapes
:��������� *
T0O
zeros_1/mul/yConst*
value	B : *
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
_output_shapes
: *
T0Q
zeros_1/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
_output_shapes
: *
T0R
zeros_1/packed/1Const*
value	B : *
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
_output_shapes
:*
T0R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
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
:(���������@D
Shape_1Shapetranspose:y:0*
_output_shapes
:*
T0_
strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
_output_shapes
: *
T0*
Index0*
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
valueB :
���������*
dtype0*
_output_shapes
: �
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
element_dtype0*
_output_shapes
: *

shape_type0�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"����@   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
element_dtype0*
_output_shapes
: *

shape_type0_
strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB: a
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
:���������@*
Index0*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: x
split/ReadVariableOpReadVariableOp"split_readvariableop_lstm_1_kernel*
_output_shapes
:	@�*
dtype0�
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
	num_split*<
_output_shapes*
(:@ :@ :@ :@ *
T0l
MatMulMatMulstrided_slice_2:output:0split:output:0*'
_output_shapes
:��������� *
T0n
MatMul_1MatMulstrided_slice_2:output:0split:output:1*'
_output_shapes
:��������� *
T0n
MatMul_2MatMulstrided_slice_2:output:0split:output:2*
T0*'
_output_shapes
:��������� n
MatMul_3MatMulstrided_slice_2:output:0split:output:3*'
_output_shapes
:��������� *
T0I
Const_1Const*
_output_shapes
: *
value	B :*
dtype0S
split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: v
split_1/ReadVariableOpReadVariableOp"split_1_readvariableop_lstm_1_bias*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
	num_split*,
_output_shapes
: : : : *
T0h
BiasAddBiasAddMatMul:product:0split_1:output:0*'
_output_shapes
:��������� *
T0l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*'
_output_shapes
:��������� *
T0l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:��������� l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:��������� v
ReadVariableOpReadVariableOp&readvariableop_lstm_1_recurrent_kernel*
dtype0*
_output_shapes
:	 �f
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

:  *
T0*
Index0n
MatMul_4MatMulzeros:output:0strided_slice_3:output:0*'
_output_shapes
:��������� *
T0d
addAddV2BiasAdd:output:0MatMul_4:product:0*
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
dtype0W
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:��������� Y
Add_1AddMul:z:0Const_3:output:0*'
_output_shapes
:��������� *
T0\
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
T0�
ReadVariableOp_1ReadVariableOp&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp*
dtype0*
_output_shapes
:	 �f
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

:  *
T0*
Index0n
MatMul_5MatMulzeros:output:0strided_slice_4:output:0*
T0*'
_output_shapes
:��������� h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*'
_output_shapes
:��������� *
T0L
Const_4Const*
_output_shapes
: *
valueB
 *��L>*
dtype0L
Const_5Const*
dtype0*
_output_shapes
: *
valueB
 *   ?[
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:��������� [
Add_3Add	Mul_1:z:0Const_5:output:0*
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
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:��������� e
mul_2Mulclip_by_value_1:z:0zeros_1:output:0*'
_output_shapes
:��������� *
T0�
ReadVariableOp_2ReadVariableOp&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp_1*
_output_shapes
:	 �*
dtype0f
strided_slice_5/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_5/stack_1Const*
valueB"    `   *
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
end_mask*
_output_shapes

:  *
Index0*
T0n
MatMul_6MatMulzeros:output:0strided_slice_5:output:0*
T0*'
_output_shapes
:��������� h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_4:z:0*'
_output_shapes
:��������� *
T0[
mul_3Mulclip_by_value:z:0Tanh:y:0*'
_output_shapes
:��������� *
T0V
add_5AddV2	mul_2:z:0	mul_3:z:0*'
_output_shapes
:��������� *
T0�
ReadVariableOp_3ReadVariableOp&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp_2*
dtype0*
_output_shapes
:	 �f
strided_slice_6/stackConst*
_output_shapes
:*
valueB"    `   *
dtype0h
strided_slice_6/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_6/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_slice_6StridedSliceReadVariableOp_3:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
Index0*
T0n
MatMul_7MatMulzeros:output:0strided_slice_6:output:0*
T0*'
_output_shapes
:��������� h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:��������� L
Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_4Mul	add_6:z:0Const_6:output:0*'
_output_shapes
:��������� *
T0[
Add_7Add	Mul_4:z:0Const_7:output:0*'
_output_shapes
:��������� *
T0^
clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:��������� V
clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:��������� K
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:��������� _
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*'
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
timeConst*
_output_shapes
: *
value	B : *
dtype0c
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"split_readvariableop_lstm_1_kernel"split_1_readvariableop_lstm_1_bias&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
bodyR
while_body_5782*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *
T
2*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_5781*
_num_original_outputsK
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
:��������� ^
while/Identity_5Identitywhile:output:5*'
_output_shapes
:��������� *
T0M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
_output_shapes
: *
T0M
while/Identity_8Identitywhile:output:8*
_output_shapes
: *
T0M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
_output_shapes
: *
T0�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"����    *
dtype0*
_output_shapes
:�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:(��������� h
strided_slice_7/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:a
strided_slice_7/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_7StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*'
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
:���������( �
IdentityIdentitystrided_slice_7:output:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp^while*'
_output_shapes
:��������� *
T0"
identityIdentity:output:0*6
_input_shapes%
#:���������(@:::2 
ReadVariableOpReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp2
whilewhile: : : :& "
 
_user_specified_nameinputs
�`
�
while_body_4923
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0&
"split_readvariableop_lstm_kernel_0&
"split_1_readvariableop_lstm_bias_0*
&readvariableop_lstm_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor$
 split_readvariableop_lstm_kernel$
 split_1_readvariableop_lstm_bias(
$readvariableop_lstm_recurrent_kernel��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:���������G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
dtype0*
_output_shapes
: *
value	B :x
split/ReadVariableOpReadVariableOp"split_readvariableop_lstm_kernel_0*
dtype0*
_output_shapes
:	��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
	num_split*<
_output_shapes*
(:@:@:@:@*
T0~
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:0*
T0*'
_output_shapes
:���������@�
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:1*'
_output_shapes
:���������@*
T0�
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:2*
T0*'
_output_shapes
:���������@�
MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:3*'
_output_shapes
:���������@*
T0I
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
_output_shapes
: *
value	B : *
dtype0v
split_1/ReadVariableOpReadVariableOp"split_1_readvariableop_lstm_bias_0*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split*,
_output_shapes
:@:@:@:@h
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:���������@l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*'
_output_shapes
:���������@*
T0l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*'
_output_shapes
:���������@*
T0l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:���������@v
ReadVariableOpReadVariableOp&readvariableop_lstm_recurrent_kernel_0*
dtype0*
_output_shapes
:	@�d
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"        f
strided_slice/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:f
strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:@@k
MatMul_4MatMulplaceholder_2strided_slice:output:0*
T0*'
_output_shapes
:���������@d
addAddV2BiasAdd:output:0MatMul_4:product:0*'
_output_shapes
:���������@*
T0L
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
: W
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:���������@Y
Add_1AddMul:z:0Const_3:output:0*
T0*'
_output_shapes
:���������@\
clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������@T
clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������@�
ReadVariableOp_1ReadVariableOp&readvariableop_lstm_recurrent_kernel_0^ReadVariableOp*
dtype0*
_output_shapes
:	@�f
strided_slice_2/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_1Const*
valueB"    �   *
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

:@@*
Index0*
T0m
MatMul_5MatMulplaceholder_2strided_slice_2:output:0*
T0*'
_output_shapes
:���������@h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:���������@L
Const_4Const*
_output_shapes
: *
valueB
 *��L>*
dtype0L
Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:���������@[
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*'
_output_shapes
:���������@^
clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:���������@V
clip_by_value_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*'
_output_shapes
:���������@*
T0b
mul_2Mulclip_by_value_1:z:0placeholder_3*
T0*'
_output_shapes
:���������@�
ReadVariableOp_2ReadVariableOp&readvariableop_lstm_recurrent_kernel_0^ReadVariableOp_1*
dtype0*
_output_shapes
:	@�f
strided_slice_3/stackConst*
dtype0*
_output_shapes
:*
valueB"    �   h
strided_slice_3/stack_1Const*
valueB"    �   *
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
end_mask*
_output_shapes

:@@*
T0*
Index0m
MatMul_6MatMulplaceholder_2strided_slice_3:output:0*
T0*'
_output_shapes
:���������@h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:���������@I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:���������@[
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:���������@V
add_5AddV2	mul_2:z:0	mul_3:z:0*'
_output_shapes
:���������@*
T0�
ReadVariableOp_3ReadVariableOp&readvariableop_lstm_recurrent_kernel_0^ReadVariableOp_2*
dtype0*
_output_shapes
:	@�f
strided_slice_4/stackConst*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_4/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_4StridedSliceReadVariableOp_3:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:@@m
MatMul_7MatMulplaceholder_2strided_slice_4:output:0*'
_output_shapes
:���������@*
T0h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*'
_output_shapes
:���������@*
T0L
Const_6Const*
dtype0*
_output_shapes
: *
valueB
 *��L>L
Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_4Mul	add_6:z:0Const_6:output:0*'
_output_shapes
:���������@*
T0[
Add_7Add	Mul_4:z:0Const_7:output:0*'
_output_shapes
:���������@*
T0^
clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:���������@V
clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*'
_output_shapes
:���������@*
T0K
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:���������@_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:���������@�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_5:z:0*
element_dtype0*
_output_shapes
: I
add_8/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_8AddV2placeholderadd_8/y:output:0*
T0*
_output_shapes
: I
add_9/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_9AddV2while_loop_counteradd_9/y:output:0*
_output_shapes
: *
T0�
IdentityIdentity	add_9:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_1Identitywhile_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_2Identity	add_8:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_4Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*'
_output_shapes
:���������@*
T0�

Identity_5Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:���������@"F
 split_1_readvariableop_lstm_bias"split_1_readvariableop_lstm_bias_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"N
$readvariableop_lstm_recurrent_kernel&readvariableop_lstm_recurrent_kernel_0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"F
 split_readvariableop_lstm_kernel"split_readvariableop_lstm_kernel_0"$
strided_slice_1strided_slice_1_0*Q
_input_shapes@
>: : : : :���������@:���������@: : :::2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp2 
ReadVariableOpReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp: : : : :	 :
 :  : : : : 
�!
�
D__inference_sequential_layer_call_and_return_conditional_losses_6113

inputs8
4features_statefulpartitionedcall_features_embeddings,
(lstm_statefulpartitionedcall_lstm_kernel*
&lstm_statefulpartitionedcall_lstm_bias6
2lstm_statefulpartitionedcall_lstm_recurrent_kernel0
,lstm_1_statefulpartitionedcall_lstm_1_kernel.
*lstm_1_statefulpartitionedcall_lstm_1_bias:
6lstm_1_statefulpartitionedcall_lstm_1_recurrent_kernel.
*dense_statefulpartitionedcall_dense_kernel,
(dense_statefulpartitionedcall_dense_bias2
.dense_1_statefulpartitionedcall_dense_1_kernel0
,dense_1_statefulpartitionedcall_dense_1_bias
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dropout/StatefulPartitionedCall� features/StatefulPartitionedCall�lstm/StatefulPartitionedCall�lstm_1/StatefulPartitionedCall�
 features/StatefulPartitionedCallStatefulPartitionedCallinputs4features_statefulpartitionedcall_features_embeddings*
Tout
2**
config_proto

GPU 

CPU2J 8*+
_output_shapes
:���������(*
Tin
2*+
_gradient_op_typePartitionedCall-4788*K
fFRD
B__inference_features_layer_call_and_return_conditional_losses_4781�
lstm/StatefulPartitionedCallStatefulPartitionedCall)features/StatefulPartitionedCall:output:0(lstm_statefulpartitionedcall_lstm_kernel&lstm_statefulpartitionedcall_lstm_bias2lstm_statefulpartitionedcall_lstm_recurrent_kernel*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:���������(@*+
_gradient_op_typePartitionedCall-5358*G
fBR@
>__inference_lstm_layer_call_and_return_conditional_losses_5076�
lstm_1/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0,lstm_1_statefulpartitionedcall_lstm_1_kernel*lstm_1_statefulpartitionedcall_lstm_1_bias6lstm_1_statefulpartitionedcall_lstm_1_recurrent_kernel*
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
_gradient_op_typePartitionedCall-5938*I
fDRB
@__inference_lstm_1_layer_call_and_return_conditional_losses_5656�
dense/StatefulPartitionedCallStatefulPartitionedCall'lstm_1/StatefulPartitionedCall:output:0*dense_statefulpartitionedcall_dense_kernel(dense_statefulpartitionedcall_dense_bias*+
_gradient_op_typePartitionedCall-5976*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_5969*
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
:����������
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:���������*+
_gradient_op_typePartitionedCall-6020*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_6008�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0.dense_1_statefulpartitionedcall_dense_1_kernel,dense_1_statefulpartitionedcall_dense_1_bias*'
_output_shapes
:���������*
Tin
2*+
_gradient_op_typePartitionedCall-6052*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_6045*
Tout
2**
config_proto

GPU 

CPU2J 8�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall!^features/StatefulPartitionedCall^lstm/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*R
_input_shapesA
?:���������(:::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2D
 features/StatefulPartitionedCall features/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : 
Ă
�
@__inference_lstm_1_layer_call_and_return_conditional_losses_8827
inputs_0&
"split_readvariableop_lstm_1_kernel&
"split_1_readvariableop_lstm_1_bias*
&readvariableop_lstm_1_recurrent_kernel
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOp�while=
ShapeShapeinputs_0*
_output_shapes
:*
T0]
strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0_
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
zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� O
zeros_1/mul/yConst*
_output_shapes
: *
value	B : *
dtype0c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
_output_shapes
: *
T0Q
zeros_1/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B : *
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*'
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
 :������������������@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB: a
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
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"����@   *
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
:���������@*
Index0*
T0G
ConstConst*
_output_shapes
: *
value	B :*
dtype0Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: x
split/ReadVariableOpReadVariableOp"split_readvariableop_lstm_1_kernel*
dtype0*
_output_shapes
:	@��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
	num_split*<
_output_shapes*
(:@ :@ :@ :@ *
T0l
MatMulMatMulstrided_slice_2:output:0split:output:0*
T0*'
_output_shapes
:��������� n
MatMul_1MatMulstrided_slice_2:output:0split:output:1*
T0*'
_output_shapes
:��������� n
MatMul_2MatMulstrided_slice_2:output:0split:output:2*
T0*'
_output_shapes
:��������� n
MatMul_3MatMulstrided_slice_2:output:0split:output:3*
T0*'
_output_shapes
:��������� I
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: v
split_1/ReadVariableOpReadVariableOp"split_1_readvariableop_lstm_1_bias*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split*,
_output_shapes
: : : : h
BiasAddBiasAddMatMul:product:0split_1:output:0*'
_output_shapes
:��������� *
T0l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:��������� l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:��������� l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*'
_output_shapes
:��������� *
T0v
ReadVariableOpReadVariableOp&readvariableop_lstm_1_recurrent_kernel*
dtype0*
_output_shapes
:	 �f
strided_slice_3/stackConst*
_output_shapes
:*
valueB"        *
dtype0h
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

:  *
T0*
Index0n
MatMul_4MatMulzeros:output:0strided_slice_3:output:0*
T0*'
_output_shapes
:��������� d
addAddV2BiasAdd:output:0MatMul_4:product:0*
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
: W
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:��������� Y
Add_1AddMul:z:0Const_3:output:0*
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
:��������� �
ReadVariableOp_1ReadVariableOp&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp*
dtype0*
_output_shapes
:	 �f
strided_slice_4/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
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

:  *
Index0*
T0n
MatMul_5MatMulzeros:output:0strided_slice_4:output:0*'
_output_shapes
:��������� *
T0h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*'
_output_shapes
:��������� *
T0L
Const_4Const*
dtype0*
_output_shapes
: *
valueB
 *��L>L
Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:��������� [
Add_3Add	Mul_1:z:0Const_5:output:0*
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
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:��������� e
mul_2Mulclip_by_value_1:z:0zeros_1:output:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_2ReadVariableOp&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp_1*
dtype0*
_output_shapes
:	 �f
strided_slice_5/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_5/stack_1Const*
valueB"    `   *
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
end_mask*
_output_shapes

:  n
MatMul_6MatMulzeros:output:0strided_slice_5:output:0*
T0*'
_output_shapes
:��������� h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*'
_output_shapes
:��������� *
T0I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:��������� [
mul_3Mulclip_by_value:z:0Tanh:y:0*'
_output_shapes
:��������� *
T0V
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_3ReadVariableOp&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp_2*
dtype0*
_output_shapes
:	 �f
strided_slice_6/stackConst*
valueB"    `   *
dtype0*
_output_shapes
:h
strided_slice_6/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_6/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_6StridedSliceReadVariableOp_3:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
end_mask*
_output_shapes

:  *
T0*
Index0*

begin_maskn
MatMul_7MatMulzeros:output:0strided_slice_6:output:0*'
_output_shapes
:��������� *
T0h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:��������� L
Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:��������� [
Add_7Add	Mul_4:z:0Const_7:output:0*'
_output_shapes
:��������� *
T0^
clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*'
_output_shapes
:��������� *
T0V
clip_by_value_2/yConst*
_output_shapes
: *
valueB
 *    *
dtype0�
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:��������� K
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:��������� _
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*'
_output_shapes
:��������� *
T0n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"split_readvariableop_lstm_1_kernel"split_1_readvariableop_lstm_1_bias&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T
2*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_8673*
_num_original_outputs*
bodyR
while_body_8674*L
_output_shapes:
8: : : : :��������� :��������� : : : : : K
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
T0^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:��������� M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
_output_shapes
: *
T0M
while/Identity_8Identitywhile:output:8*
_output_shapes
: *
T0M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
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
strided_slice_7/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:a
strided_slice_7/stack_1Const*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_7/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
strided_slice_7StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
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
T0�
IdentityIdentitystrided_slice_7:output:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp^while*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*?
_input_shapes.
,:������������������@:::2 
ReadVariableOpReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp2
whilewhile:( $
"
_user_specified_name
inputs/0: : : 
�
�
while_cond_5502
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
lstm_1_kernel
lstm_1_bias
lstm_1_recurrent_kernel
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
identityIdentity:output:0*Q
_input_shapes@
>: : : : :��������� :��������� : : :::: : : :	 :
 :  : : : : : 
�B
�
@__inference_lstm_1_layer_call_and_return_conditional_losses_4761

inputs)
%statefulpartitionedcall_lstm_1_kernel'
#statefulpartitionedcall_lstm_1_bias3
/statefulpartitionedcall_lstm_1_recurrent_kernel
identity��StatefulPartitionedCall�while;
ShapeShapeinputs*
_output_shapes
:*
T0]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: _
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
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
_output_shapes
:*
T0*
NP
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� O
zeros_1/mul/yConst*
dtype0*
_output_shapes
: *
value	B : c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B : *
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*!
valueB"          *
dtype0v
	transpose	Transposeinputstranspose/perm:output:0*4
_output_shapes"
 :������������������@*
T0D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB: a
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
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: f
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
valueB"����@   *
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
strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:���������@�
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0%statefulpartitionedcall_lstm_1_kernel#statefulpartitionedcall_lstm_1_bias/statefulpartitionedcall_lstm_1_recurrent_kernel*+
_gradient_op_typePartitionedCall-4271*N
fIRG
E__inference_lstm_cell_1_layer_call_and_return_conditional_losses_4247*
Tout
2**
config_proto

GPU 

CPU2J 8*M
_output_shapes;
9:��������� :��������� :��������� *
Tin

2n
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
timeConst*
_output_shapes
: *
value	B : *
dtype0c
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0%statefulpartitionedcall_lstm_1_kernel#statefulpartitionedcall_lstm_1_bias/statefulpartitionedcall_lstm_1_recurrent_kernel^StatefulPartitionedCall*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *
T
2*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_4684*
_num_original_outputs*
bodyR
while_body_4685K
while/IdentityIdentitywhile:output:0*
_output_shapes
: *
T0M
while/Identity_1Identitywhile:output:1*
_output_shapes
: *
T0M
while/Identity_2Identitywhile:output:2*
_output_shapes
: *
T0M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*'
_output_shapes
:��������� *
T0^
while/Identity_5Identitywhile:output:5*'
_output_shapes
:��������� *
T0M
while/Identity_6Identitywhile:output:6*
_output_shapes
: *
T0M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
_output_shapes
: *
T0M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
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
strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
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
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*4
_output_shapes"
 :������������������ *
T0�
IdentityIdentitystrided_slice_3:output:0^StatefulPartitionedCall^while*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*?
_input_shapes.
,:������������������@:::2
whilewhile22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
�
�
$__inference_dense_layer_call_fn_9714

inputs(
$statefulpartitionedcall_dense_kernel&
"statefulpartitionedcall_dense_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs$statefulpartitionedcall_dense_kernel"statefulpartitionedcall_dense_bias*+
_gradient_op_typePartitionedCall-5976*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_5969*
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
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:��������� ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�L
�
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_10159

inputs
states_0
states_1&
"split_readvariableop_lstm_1_kernel&
"split_1_readvariableop_lstm_1_bias*
&readvariableop_lstm_1_recurrent_kernel
identity

identity_1

identity_2��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOpG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: x
split/ReadVariableOpReadVariableOp"split_readvariableop_lstm_1_kernel*
dtype0*
_output_shapes
:	@��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split*<
_output_shapes*
(:@ :@ :@ :@ Z
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:��������� \
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:��������� \
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:��������� \
MatMul_3MatMulinputssplit:output:3*'
_output_shapes
:��������� *
T0I
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: v
split_1/ReadVariableOpReadVariableOp"split_1_readvariableop_lstm_1_bias*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split*,
_output_shapes
: : : : h
BiasAddBiasAddMatMul:product:0split_1:output:0*'
_output_shapes
:��������� *
T0l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:��������� l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:��������� l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:��������� v
ReadVariableOpReadVariableOp&readvariableop_lstm_1_recurrent_kernel*
dtype0*
_output_shapes
:	 �d
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
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
Index0*
T0f
MatMul_4MatMulstates_0strided_slice:output:0*
T0*'
_output_shapes
:��������� d
addAddV2BiasAdd:output:0MatMul_4:product:0*
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
: W
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:��������� Y
Add_1AddMul:z:0Const_3:output:0*
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
clip_by_value/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_1ReadVariableOp&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp*
dtype0*
_output_shapes
:	 �f
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
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
_output_shapes

:  *
T0*
Index0*

begin_mask*
end_maskh
MatMul_5MatMulstates_0strided_slice_1:output:0*'
_output_shapes
:��������� *
T0h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:��������� L
Const_4Const*
dtype0*
_output_shapes
: *
valueB
 *��L>L
Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:��������� [
Add_3Add	Mul_1:z:0Const_5:output:0*
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
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:��������� ]
mul_2Mulclip_by_value_1:z:0states_1*
T0*'
_output_shapes
:��������� �
ReadVariableOp_2ReadVariableOp&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp_1*
dtype0*
_output_shapes
:	 �f
strided_slice_2/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_1Const*
valueB"    `   *
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
end_mask*
_output_shapes

:  *
T0*
Index0h
MatMul_6MatMulstates_0strided_slice_2:output:0*
T0*'
_output_shapes
:��������� h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*'
_output_shapes
:��������� *
T0I
TanhTanh	add_4:z:0*'
_output_shapes
:��������� *
T0[
mul_3Mulclip_by_value:z:0Tanh:y:0*'
_output_shapes
:��������� *
T0V
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:��������� �
ReadVariableOp_3ReadVariableOp&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp_2*
dtype0*
_output_shapes
:	 �f
strided_slice_3/stackConst*
valueB"    `   *
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
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
T0*
Index0h
MatMul_7MatMulstates_0strided_slice_3:output:0*'
_output_shapes
:��������� *
T0h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:��������� L
Const_6Const*
dtype0*
_output_shapes
: *
valueB
 *��L>L
Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:��������� [
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:��������� ^
clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:��������� V
clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:��������� K
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:��������� _
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:��������� �
IdentityIdentity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:��������� �

Identity_1Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*'
_output_shapes
:��������� *
T0�

Identity_2Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*X
_input_shapesG
E:���������@:��������� :��������� :::20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp2 
ReadVariableOpReadVariableOp: : : :& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0:($
"
_user_specified_name
states/1
��
�
>__inference_lstm_layer_call_and_return_conditional_losses_8253

inputs$
 split_readvariableop_lstm_kernel$
 split_1_readvariableop_lstm_bias(
$readvariableop_lstm_recurrent_kernel
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOp�while;
ShapeShapeinputs*
_output_shapes
:*
T0]
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
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :@*
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
value	B :@*
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
:���������@O
zeros_1/mul/yConst*
dtype0*
_output_shapes
: *
value	B :@c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
_output_shapes
: *
T0Q
zeros_1/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
_output_shapes
:*
T0R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:(���������D
Shape_1Shapetranspose:y:0*
_output_shapes
:*
T0_
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
strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:���������*
Index0*
T0G
ConstConst*
_output_shapes
: *
value	B :*
dtype0Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: v
split/ReadVariableOpReadVariableOp split_readvariableop_lstm_kernel*
dtype0*
_output_shapes
:	��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
	num_split*<
_output_shapes*
(:@:@:@:@*
T0l
MatMulMatMulstrided_slice_2:output:0split:output:0*'
_output_shapes
:���������@*
T0n
MatMul_1MatMulstrided_slice_2:output:0split:output:1*'
_output_shapes
:���������@*
T0n
MatMul_2MatMulstrided_slice_2:output:0split:output:2*'
_output_shapes
:���������@*
T0n
MatMul_3MatMulstrided_slice_2:output:0split:output:3*'
_output_shapes
:���������@*
T0I
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: t
split_1/ReadVariableOpReadVariableOp split_1_readvariableop_lstm_bias*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split*,
_output_shapes
:@:@:@:@h
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:���������@l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:���������@l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:���������@l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:���������@t
ReadVariableOpReadVariableOp$readvariableop_lstm_recurrent_kernel*
dtype0*
_output_shapes
:	@�f
strided_slice_3/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_3/stack_1Const*
valueB"    @   *
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

:@@*
Index0*
T0n
MatMul_4MatMulzeros:output:0strided_slice_3:output:0*'
_output_shapes
:���������@*
T0d
addAddV2BiasAdd:output:0MatMul_4:product:0*'
_output_shapes
:���������@*
T0L
Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *��L>L
Const_3Const*
valueB
 *   ?*
dtype0*
_output_shapes
: W
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:���������@Y
Add_1AddMul:z:0Const_3:output:0*
T0*'
_output_shapes
:���������@\
clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������@T
clip_by_value/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*'
_output_shapes
:���������@*
T0�
ReadVariableOp_1ReadVariableOp$readvariableop_lstm_recurrent_kernel^ReadVariableOp*
dtype0*
_output_shapes
:	@�f
strided_slice_4/stackConst*
dtype0*
_output_shapes
:*
valueB"    @   h
strided_slice_4/stack_1Const*
valueB"    �   *
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

:@@*
T0*
Index0n
MatMul_5MatMulzeros:output:0strided_slice_4:output:0*
T0*'
_output_shapes
:���������@h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:���������@L
Const_4Const*
_output_shapes
: *
valueB
 *��L>*
dtype0L
Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:���������@[
Add_3Add	Mul_1:z:0Const_5:output:0*'
_output_shapes
:���������@*
T0^
clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*'
_output_shapes
:���������@*
T0V
clip_by_value_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*'
_output_shapes
:���������@*
T0e
mul_2Mulclip_by_value_1:z:0zeros_1:output:0*
T0*'
_output_shapes
:���������@�
ReadVariableOp_2ReadVariableOp$readvariableop_lstm_recurrent_kernel^ReadVariableOp_1*
_output_shapes
:	@�*
dtype0f
strided_slice_5/stackConst*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_5/stack_1Const*
valueB"    �   *
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
end_mask*
_output_shapes

:@@*
T0*
Index0n
MatMul_6MatMulzeros:output:0strided_slice_5:output:0*
T0*'
_output_shapes
:���������@h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:���������@I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:���������@[
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:���������@V
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:���������@�
ReadVariableOp_3ReadVariableOp$readvariableop_lstm_recurrent_kernel^ReadVariableOp_2*
_output_shapes
:	@�*
dtype0f
strided_slice_6/stackConst*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_6/stack_1Const*
dtype0*
_output_shapes
:*
valueB"        h
strided_slice_6/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_slice_6StridedSliceReadVariableOp_3:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:@@*
T0*
Index0n
MatMul_7MatMulzeros:output:0strided_slice_6:output:0*
T0*'
_output_shapes
:���������@h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:���������@L
Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:���������@[
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:���������@^
clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*'
_output_shapes
:���������@*
T0V
clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:���������@K
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:���������@_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*'
_output_shapes
:���������@*
T0n
TensorArrayV2_1/element_shapeConst*
valueB"����@   *
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
while/maximum_iterationsConst*
dtype0*
_output_shapes
: *
valueB :
���������T
while/loop_counterConst*
dtype0*
_output_shapes
: *
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 split_readvariableop_lstm_kernel split_1_readvariableop_lstm_bias$readvariableop_lstm_recurrent_kernel^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
bodyR
while_body_8100*L
_output_shapes:
8: : : : :���������@:���������@: : : : : *K
output_shapes:
8: : : : :���������@:���������@: : : : : *
T
2*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_8099*
_num_original_outputsK
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
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:���������@^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:���������@M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
_output_shapes
: *
T0M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"����@   *
dtype0*
_output_shapes
:�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:(���������@h
strided_slice_7/stackConst*
_output_shapes
:*
valueB:
���������*
dtype0a
strided_slice_7/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_7StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*'
_output_shapes
:���������@e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������(@�
IdentityIdentitytranspose_1:y:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp^while*
T0*+
_output_shapes
:���������(@"
identityIdentity:output:0*6
_input_shapes%
#:���������(:::20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp2
whilewhile2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs: : : 
��
�
D__inference_sequential_layer_call_and_return_conditional_losses_6779

inputs1
-features_embedding_lookup_features_embeddings)
%lstm_split_readvariableop_lstm_kernel)
%lstm_split_1_readvariableop_lstm_bias-
)lstm_readvariableop_lstm_recurrent_kernel-
)lstm_1_split_readvariableop_lstm_1_kernel-
)lstm_1_split_1_readvariableop_lstm_1_bias1
-lstm_1_readvariableop_lstm_1_recurrent_kernel,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias0
,dense_1_matmul_readvariableop_dense_1_kernel/
+dense_1_biasadd_readvariableop_dense_1_bias
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�features/embedding_lookup�lstm/ReadVariableOp�lstm/ReadVariableOp_1�lstm/ReadVariableOp_2�lstm/ReadVariableOp_3�lstm/split/ReadVariableOp�lstm/split_1/ReadVariableOp�
lstm/while�lstm_1/ReadVariableOp�lstm_1/ReadVariableOp_1�lstm_1/ReadVariableOp_2�lstm_1/ReadVariableOp_3�lstm_1/split/ReadVariableOp�lstm_1/split_1/ReadVariableOp�lstm_1/while^
features/CastCastinputs*

SrcT0*

DstT0*'
_output_shapes
:���������(�
features/embedding_lookupResourceGather-features_embedding_lookup_features_embeddingsfeatures/Cast:y:0*
Tindices0*
dtype0*+
_output_shapes
:���������(*@
_class6
42loc:@features/embedding_lookup/features/embeddings�
"features/embedding_lookup/IdentityIdentity"features/embedding_lookup:output:0*
T0*@
_class6
42loc:@features/embedding_lookup/features/embeddings*+
_output_shapes
:���������(�
$features/embedding_lookup/Identity_1Identity+features/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������(g

lstm/ShapeShape-features/embedding_lookup/Identity_1:output:0*
_output_shapes
:*
T0b
lstm/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:d
lstm/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0d
lstm/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: R
lstm/zeros/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: n
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
_output_shapes
: *
T0T
lstm/zeros/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: h
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: U
lstm/zeros/packed/1Const*
_output_shapes
: *
value	B :@*
dtype0�
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
_output_shapes
:*
T0U
lstm/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    {

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:���������@T
lstm/zeros_1/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: r
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
_output_shapes
: *
T0V
lstm/zeros_1/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: n
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
_output_shapes
: *
T0W
lstm/zeros_1/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: �
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:W
lstm/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: �
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*'
_output_shapes
:���������@*
T0h
lstm/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
lstm/transpose	Transpose-features/embedding_lookup/Identity_1:output:0lstm/transpose/perm:output:0*
T0*+
_output_shapes
:(���������N
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:d
lstm/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:f
lstm/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:f
lstm/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_maskk
 lstm/TensorArrayV2/element_shapeConst*
valueB :
���������*
dtype0*
_output_shapes
: �
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: �
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: d
lstm/strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:f
lstm/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:f
lstm/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:���������*
Index0*
T0L

lstm/ConstConst*
value	B :*
dtype0*
_output_shapes
: V
lstm/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: �
lstm/split/ReadVariableOpReadVariableOp%lstm_split_readvariableop_lstm_kernel*
dtype0*
_output_shapes
:	��

lstm/splitSplitlstm/split/split_dim:output:0!lstm/split/ReadVariableOp:value:0*
T0*
	num_split*<
_output_shapes*
(:@:@:@:@{
lstm/MatMulMatMullstm/strided_slice_2:output:0lstm/split:output:0*'
_output_shapes
:���������@*
T0}
lstm/MatMul_1MatMullstm/strided_slice_2:output:0lstm/split:output:1*'
_output_shapes
:���������@*
T0}
lstm/MatMul_2MatMullstm/strided_slice_2:output:0lstm/split:output:2*
T0*'
_output_shapes
:���������@}
lstm/MatMul_3MatMullstm/strided_slice_2:output:0lstm/split:output:3*
T0*'
_output_shapes
:���������@N
lstm/Const_1Const*
_output_shapes
: *
value	B :*
dtype0X
lstm/split_1/split_dimConst*
_output_shapes
: *
value	B : *
dtype0~
lstm/split_1/ReadVariableOpReadVariableOp%lstm_split_1_readvariableop_lstm_bias*
dtype0*
_output_shapes	
:��
lstm/split_1Splitlstm/split_1/split_dim:output:0#lstm/split_1/ReadVariableOp:value:0*
	num_split*,
_output_shapes
:@:@:@:@*
T0w
lstm/BiasAddBiasAddlstm/MatMul:product:0lstm/split_1:output:0*'
_output_shapes
:���������@*
T0{
lstm/BiasAdd_1BiasAddlstm/MatMul_1:product:0lstm/split_1:output:1*'
_output_shapes
:���������@*
T0{
lstm/BiasAdd_2BiasAddlstm/MatMul_2:product:0lstm/split_1:output:2*
T0*'
_output_shapes
:���������@{
lstm/BiasAdd_3BiasAddlstm/MatMul_3:product:0lstm/split_1:output:3*
T0*'
_output_shapes
:���������@~
lstm/ReadVariableOpReadVariableOp)lstm_readvariableop_lstm_recurrent_kernel*
dtype0*
_output_shapes
:	@�k
lstm/strided_slice_3/stackConst*
valueB"        *
dtype0*
_output_shapes
:m
lstm/strided_slice_3/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:m
lstm/strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
lstm/strided_slice_3StridedSlicelstm/ReadVariableOp:value:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:@@*
T0*
Index0}
lstm/MatMul_4MatMullstm/zeros:output:0lstm/strided_slice_3:output:0*
T0*'
_output_shapes
:���������@s
lstm/addAddV2lstm/BiasAdd:output:0lstm/MatMul_4:product:0*
T0*'
_output_shapes
:���������@Q
lstm/Const_2Const*
valueB
 *��L>*
dtype0*
_output_shapes
: Q
lstm/Const_3Const*
valueB
 *   ?*
dtype0*
_output_shapes
: f
lstm/MulMullstm/add:z:0lstm/Const_2:output:0*'
_output_shapes
:���������@*
T0h

lstm/Add_1Addlstm/Mul:z:0lstm/Const_3:output:0*'
_output_shapes
:���������@*
T0a
lstm/clip_by_value/Minimum/yConst*
dtype0*
_output_shapes
: *
valueB
 *  �?�
lstm/clip_by_value/MinimumMinimumlstm/Add_1:z:0%lstm/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������@Y
lstm/clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
lstm/clip_by_valueMaximumlstm/clip_by_value/Minimum:z:0lstm/clip_by_value/y:output:0*
T0*'
_output_shapes
:���������@�
lstm/ReadVariableOp_1ReadVariableOp)lstm_readvariableop_lstm_recurrent_kernel^lstm/ReadVariableOp*
dtype0*
_output_shapes
:	@�k
lstm/strided_slice_4/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:m
lstm/strided_slice_4/stack_1Const*
_output_shapes
:*
valueB"    �   *
dtype0m
lstm/strided_slice_4/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
lstm/strided_slice_4StridedSlicelstm/ReadVariableOp_1:value:0#lstm/strided_slice_4/stack:output:0%lstm/strided_slice_4/stack_1:output:0%lstm/strided_slice_4/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:@@}
lstm/MatMul_5MatMullstm/zeros:output:0lstm/strided_slice_4:output:0*
T0*'
_output_shapes
:���������@w

lstm/add_2AddV2lstm/BiasAdd_1:output:0lstm/MatMul_5:product:0*'
_output_shapes
:���������@*
T0Q
lstm/Const_4Const*
_output_shapes
: *
valueB
 *��L>*
dtype0Q
lstm/Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: j

lstm/Mul_1Mullstm/add_2:z:0lstm/Const_4:output:0*'
_output_shapes
:���������@*
T0j

lstm/Add_3Addlstm/Mul_1:z:0lstm/Const_5:output:0*'
_output_shapes
:���������@*
T0c
lstm/clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
lstm/clip_by_value_1/MinimumMinimumlstm/Add_3:z:0'lstm/clip_by_value_1/Minimum/y:output:0*'
_output_shapes
:���������@*
T0[
lstm/clip_by_value_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
lstm/clip_by_value_1Maximum lstm/clip_by_value_1/Minimum:z:0lstm/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:���������@t

lstm/mul_2Mullstm/clip_by_value_1:z:0lstm/zeros_1:output:0*
T0*'
_output_shapes
:���������@�
lstm/ReadVariableOp_2ReadVariableOp)lstm_readvariableop_lstm_recurrent_kernel^lstm/ReadVariableOp_1*
dtype0*
_output_shapes
:	@�k
lstm/strided_slice_5/stackConst*
dtype0*
_output_shapes
:*
valueB"    �   m
lstm/strided_slice_5/stack_1Const*
valueB"    �   *
dtype0*
_output_shapes
:m
lstm/strided_slice_5/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
lstm/strided_slice_5StridedSlicelstm/ReadVariableOp_2:value:0#lstm/strided_slice_5/stack:output:0%lstm/strided_slice_5/stack_1:output:0%lstm/strided_slice_5/stack_2:output:0*
_output_shapes

:@@*
T0*
Index0*

begin_mask*
end_mask}
lstm/MatMul_6MatMullstm/zeros:output:0lstm/strided_slice_5:output:0*
T0*'
_output_shapes
:���������@w

lstm/add_4AddV2lstm/BiasAdd_2:output:0lstm/MatMul_6:product:0*
T0*'
_output_shapes
:���������@S
	lstm/TanhTanhlstm/add_4:z:0*
T0*'
_output_shapes
:���������@j

lstm/mul_3Mullstm/clip_by_value:z:0lstm/Tanh:y:0*
T0*'
_output_shapes
:���������@e

lstm/add_5AddV2lstm/mul_2:z:0lstm/mul_3:z:0*
T0*'
_output_shapes
:���������@�
lstm/ReadVariableOp_3ReadVariableOp)lstm_readvariableop_lstm_recurrent_kernel^lstm/ReadVariableOp_2*
dtype0*
_output_shapes
:	@�k
lstm/strided_slice_6/stackConst*
valueB"    �   *
dtype0*
_output_shapes
:m
lstm/strided_slice_6/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:m
lstm/strided_slice_6/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
lstm/strided_slice_6StridedSlicelstm/ReadVariableOp_3:value:0#lstm/strided_slice_6/stack:output:0%lstm/strided_slice_6/stack_1:output:0%lstm/strided_slice_6/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:@@}
lstm/MatMul_7MatMullstm/zeros:output:0lstm/strided_slice_6:output:0*'
_output_shapes
:���������@*
T0w

lstm/add_6AddV2lstm/BiasAdd_3:output:0lstm/MatMul_7:product:0*
T0*'
_output_shapes
:���������@Q
lstm/Const_6Const*
_output_shapes
: *
valueB
 *��L>*
dtype0Q
lstm/Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: j

lstm/Mul_4Mullstm/add_6:z:0lstm/Const_6:output:0*'
_output_shapes
:���������@*
T0j

lstm/Add_7Addlstm/Mul_4:z:0lstm/Const_7:output:0*
T0*'
_output_shapes
:���������@c
lstm/clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
lstm/clip_by_value_2/MinimumMinimumlstm/Add_7:z:0'lstm/clip_by_value_2/Minimum/y:output:0*'
_output_shapes
:���������@*
T0[
lstm/clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
lstm/clip_by_value_2Maximum lstm/clip_by_value_2/Minimum:z:0lstm/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:���������@U
lstm/Tanh_1Tanhlstm/add_5:z:0*
T0*'
_output_shapes
:���������@n

lstm/mul_5Mullstm/clip_by_value_2:z:0lstm/Tanh_1:y:0*
T0*'
_output_shapes
:���������@s
"lstm/TensorArrayV2_1/element_shapeConst*
valueB"����@   *
dtype0*
_output_shapes
:�
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: K
	lstm/timeConst*
value	B : *
dtype0*
_output_shapes
: h
lstm/while/maximum_iterationsConst*
_output_shapes
: *
valueB :
���������*
dtype0Y
lstm/while/loop_counterConst*
_output_shapes
: *
value	B : *
dtype0�

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0%lstm_split_readvariableop_lstm_kernel%lstm_split_1_readvariableop_lstm_bias)lstm_readvariableop_lstm_recurrent_kernel^lstm/ReadVariableOp_3^lstm/split/ReadVariableOp^lstm/split_1/ReadVariableOp* 
bodyR
lstm_while_body_6321*L
_output_shapes:
8: : : : :���������@:���������@: : : : : *
T
2*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
_lower_using_switch_merge(*
parallel_iterations * 
condR
lstm_while_cond_6320*
_num_original_outputsU
lstm/while/IdentityIdentitylstm/while:output:0*
T0*
_output_shapes
: W
lstm/while/Identity_1Identitylstm/while:output:1*
_output_shapes
: *
T0W
lstm/while/Identity_2Identitylstm/while:output:2*
T0*
_output_shapes
: W
lstm/while/Identity_3Identitylstm/while:output:3*
T0*
_output_shapes
: h
lstm/while/Identity_4Identitylstm/while:output:4*'
_output_shapes
:���������@*
T0h
lstm/while/Identity_5Identitylstm/while:output:5*'
_output_shapes
:���������@*
T0W
lstm/while/Identity_6Identitylstm/while:output:6*
_output_shapes
: *
T0W
lstm/while/Identity_7Identitylstm/while:output:7*
T0*
_output_shapes
: W
lstm/while/Identity_8Identitylstm/while:output:8*
T0*
_output_shapes
: W
lstm/while/Identity_9Identitylstm/while:output:9*
T0*
_output_shapes
: Y
lstm/while/Identity_10Identitylstm/while:output:10*
_output_shapes
: *
T0�
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"����@   *
dtype0*
_output_shapes
:�
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while/Identity_3:output:0>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:(���������@m
lstm/strided_slice_7/stackConst*
dtype0*
_output_shapes
:*
valueB:
���������f
lstm/strided_slice_7/stack_1Const*
valueB: *
dtype0*
_output_shapes
:f
lstm/strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
lstm/strided_slice_7StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_7/stack:output:0%lstm/strided_slice_7/stack_1:output:0%lstm/strided_slice_7/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:���������@j
lstm/transpose_1/permConst*
_output_shapes
:*!
valueB"          *
dtype0�
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������(@P
lstm_1/ShapeShapelstm/transpose_1:y:0*
_output_shapes
:*
T0d
lstm_1/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0f
lstm_1/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:f
lstm_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
lstm_1/strided_sliceStridedSlicelstm_1/Shape:output:0#lstm_1/strided_slice/stack:output:0%lstm_1/strided_slice/stack_1:output:0%lstm_1/strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0T
lstm_1/zeros/mul/yConst*
_output_shapes
: *
value	B : *
dtype0t
lstm_1/zeros/mulMullstm_1/strided_slice:output:0lstm_1/zeros/mul/y:output:0*
_output_shapes
: *
T0V
lstm_1/zeros/Less/yConst*
_output_shapes
: *
value
B :�*
dtype0n
lstm_1/zeros/LessLesslstm_1/zeros/mul:z:0lstm_1/zeros/Less/y:output:0*
T0*
_output_shapes
: W
lstm_1/zeros/packed/1Const*
value	B : *
dtype0*
_output_shapes
: �
lstm_1/zeros/packedPacklstm_1/strided_slice:output:0lstm_1/zeros/packed/1:output:0*
_output_shapes
:*
T0*
NW
lstm_1/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0�
lstm_1/zerosFilllstm_1/zeros/packed:output:0lstm_1/zeros/Const:output:0*
T0*'
_output_shapes
:��������� V
lstm_1/zeros_1/mul/yConst*
value	B : *
dtype0*
_output_shapes
: x
lstm_1/zeros_1/mulMullstm_1/strided_slice:output:0lstm_1/zeros_1/mul/y:output:0*
_output_shapes
: *
T0X
lstm_1/zeros_1/Less/yConst*
_output_shapes
: *
value
B :�*
dtype0t
lstm_1/zeros_1/LessLesslstm_1/zeros_1/mul:z:0lstm_1/zeros_1/Less/y:output:0*
T0*
_output_shapes
: Y
lstm_1/zeros_1/packed/1Const*
value	B : *
dtype0*
_output_shapes
: �
lstm_1/zeros_1/packedPacklstm_1/strided_slice:output:0 lstm_1/zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:Y
lstm_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: �
lstm_1/zeros_1Filllstm_1/zeros_1/packed:output:0lstm_1/zeros_1/Const:output:0*'
_output_shapes
:��������� *
T0j
lstm_1/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
lstm_1/transpose	Transposelstm/transpose_1:y:0lstm_1/transpose/perm:output:0*
T0*+
_output_shapes
:(���������@R
lstm_1/Shape_1Shapelstm_1/transpose:y:0*
_output_shapes
:*
T0f
lstm_1/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:h
lstm_1/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:h
lstm_1/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
lstm_1/strided_slice_1StridedSlicelstm_1/Shape_1:output:0%lstm_1/strided_slice_1/stack:output:0'lstm_1/strided_slice_1/stack_1:output:0'lstm_1/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: m
"lstm_1/TensorArrayV2/element_shapeConst*
dtype0*
_output_shapes
: *
valueB :
����������
lstm_1/TensorArrayV2TensorListReserve+lstm_1/TensorArrayV2/element_shape:output:0lstm_1/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: �
<lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"����@   *
dtype0*
_output_shapes
:�
.lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_1/transpose:y:0Elstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: f
lstm_1/strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:h
lstm_1/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:h
lstm_1/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
lstm_1/strided_slice_2StridedSlicelstm_1/transpose:y:0%lstm_1/strided_slice_2/stack:output:0'lstm_1/strided_slice_2/stack_1:output:0'lstm_1/strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:���������@*
Index0*
T0N
lstm_1/ConstConst*
_output_shapes
: *
value	B :*
dtype0X
lstm_1/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: �
lstm_1/split/ReadVariableOpReadVariableOp)lstm_1_split_readvariableop_lstm_1_kernel*
dtype0*
_output_shapes
:	@��
lstm_1/splitSplitlstm_1/split/split_dim:output:0#lstm_1/split/ReadVariableOp:value:0*
T0*
	num_split*<
_output_shapes*
(:@ :@ :@ :@ �
lstm_1/MatMulMatMullstm_1/strided_slice_2:output:0lstm_1/split:output:0*'
_output_shapes
:��������� *
T0�
lstm_1/MatMul_1MatMullstm_1/strided_slice_2:output:0lstm_1/split:output:1*
T0*'
_output_shapes
:��������� �
lstm_1/MatMul_2MatMullstm_1/strided_slice_2:output:0lstm_1/split:output:2*
T0*'
_output_shapes
:��������� �
lstm_1/MatMul_3MatMullstm_1/strided_slice_2:output:0lstm_1/split:output:3*'
_output_shapes
:��������� *
T0P
lstm_1/Const_1Const*
value	B :*
dtype0*
_output_shapes
: Z
lstm_1/split_1/split_dimConst*
_output_shapes
: *
value	B : *
dtype0�
lstm_1/split_1/ReadVariableOpReadVariableOp)lstm_1_split_1_readvariableop_lstm_1_bias*
dtype0*
_output_shapes	
:��
lstm_1/split_1Split!lstm_1/split_1/split_dim:output:0%lstm_1/split_1/ReadVariableOp:value:0*
	num_split*,
_output_shapes
: : : : *
T0}
lstm_1/BiasAddBiasAddlstm_1/MatMul:product:0lstm_1/split_1:output:0*'
_output_shapes
:��������� *
T0�
lstm_1/BiasAdd_1BiasAddlstm_1/MatMul_1:product:0lstm_1/split_1:output:1*
T0*'
_output_shapes
:��������� �
lstm_1/BiasAdd_2BiasAddlstm_1/MatMul_2:product:0lstm_1/split_1:output:2*'
_output_shapes
:��������� *
T0�
lstm_1/BiasAdd_3BiasAddlstm_1/MatMul_3:product:0lstm_1/split_1:output:3*
T0*'
_output_shapes
:��������� �
lstm_1/ReadVariableOpReadVariableOp-lstm_1_readvariableop_lstm_1_recurrent_kernel*
dtype0*
_output_shapes
:	 �m
lstm_1/strided_slice_3/stackConst*
valueB"        *
dtype0*
_output_shapes
:o
lstm_1/strided_slice_3/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:o
lstm_1/strided_slice_3/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
lstm_1/strided_slice_3StridedSlicelstm_1/ReadVariableOp:value:0%lstm_1/strided_slice_3/stack:output:0'lstm_1/strided_slice_3/stack_1:output:0'lstm_1/strided_slice_3/stack_2:output:0*
_output_shapes

:  *
Index0*
T0*

begin_mask*
end_mask�
lstm_1/MatMul_4MatMullstm_1/zeros:output:0lstm_1/strided_slice_3:output:0*'
_output_shapes
:��������� *
T0y

lstm_1/addAddV2lstm_1/BiasAdd:output:0lstm_1/MatMul_4:product:0*'
_output_shapes
:��������� *
T0S
lstm_1/Const_2Const*
valueB
 *��L>*
dtype0*
_output_shapes
: S
lstm_1/Const_3Const*
_output_shapes
: *
valueB
 *   ?*
dtype0l

lstm_1/MulMullstm_1/add:z:0lstm_1/Const_2:output:0*
T0*'
_output_shapes
:��������� n
lstm_1/Add_1Addlstm_1/Mul:z:0lstm_1/Const_3:output:0*'
_output_shapes
:��������� *
T0c
lstm_1/clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
lstm_1/clip_by_value/MinimumMinimumlstm_1/Add_1:z:0'lstm_1/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:��������� [
lstm_1/clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
lstm_1/clip_by_valueMaximum lstm_1/clip_by_value/Minimum:z:0lstm_1/clip_by_value/y:output:0*
T0*'
_output_shapes
:��������� �
lstm_1/ReadVariableOp_1ReadVariableOp-lstm_1_readvariableop_lstm_1_recurrent_kernel^lstm_1/ReadVariableOp*
dtype0*
_output_shapes
:	 �m
lstm_1/strided_slice_4/stackConst*
dtype0*
_output_shapes
:*
valueB"        o
lstm_1/strided_slice_4/stack_1Const*
dtype0*
_output_shapes
:*
valueB"    @   o
lstm_1/strided_slice_4/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
lstm_1/strided_slice_4StridedSlicelstm_1/ReadVariableOp_1:value:0%lstm_1/strided_slice_4/stack:output:0'lstm_1/strided_slice_4/stack_1:output:0'lstm_1/strided_slice_4/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:  *
Index0*
T0�
lstm_1/MatMul_5MatMullstm_1/zeros:output:0lstm_1/strided_slice_4:output:0*
T0*'
_output_shapes
:��������� }
lstm_1/add_2AddV2lstm_1/BiasAdd_1:output:0lstm_1/MatMul_5:product:0*'
_output_shapes
:��������� *
T0S
lstm_1/Const_4Const*
valueB
 *��L>*
dtype0*
_output_shapes
: S
lstm_1/Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: p
lstm_1/Mul_1Mullstm_1/add_2:z:0lstm_1/Const_4:output:0*'
_output_shapes
:��������� *
T0p
lstm_1/Add_3Addlstm_1/Mul_1:z:0lstm_1/Const_5:output:0*
T0*'
_output_shapes
:��������� e
 lstm_1/clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
lstm_1/clip_by_value_1/MinimumMinimumlstm_1/Add_3:z:0)lstm_1/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:��������� ]
lstm_1/clip_by_value_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
lstm_1/clip_by_value_1Maximum"lstm_1/clip_by_value_1/Minimum:z:0!lstm_1/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:��������� z
lstm_1/mul_2Mullstm_1/clip_by_value_1:z:0lstm_1/zeros_1:output:0*
T0*'
_output_shapes
:��������� �
lstm_1/ReadVariableOp_2ReadVariableOp-lstm_1_readvariableop_lstm_1_recurrent_kernel^lstm_1/ReadVariableOp_1*
dtype0*
_output_shapes
:	 �m
lstm_1/strided_slice_5/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:o
lstm_1/strided_slice_5/stack_1Const*
_output_shapes
:*
valueB"    `   *
dtype0o
lstm_1/strided_slice_5/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
lstm_1/strided_slice_5StridedSlicelstm_1/ReadVariableOp_2:value:0%lstm_1/strided_slice_5/stack:output:0'lstm_1/strided_slice_5/stack_1:output:0'lstm_1/strided_slice_5/stack_2:output:0*
end_mask*
_output_shapes

:  *
T0*
Index0*

begin_mask�
lstm_1/MatMul_6MatMullstm_1/zeros:output:0lstm_1/strided_slice_5:output:0*
T0*'
_output_shapes
:��������� }
lstm_1/add_4AddV2lstm_1/BiasAdd_2:output:0lstm_1/MatMul_6:product:0*'
_output_shapes
:��������� *
T0W
lstm_1/TanhTanhlstm_1/add_4:z:0*
T0*'
_output_shapes
:��������� p
lstm_1/mul_3Mullstm_1/clip_by_value:z:0lstm_1/Tanh:y:0*
T0*'
_output_shapes
:��������� k
lstm_1/add_5AddV2lstm_1/mul_2:z:0lstm_1/mul_3:z:0*
T0*'
_output_shapes
:��������� �
lstm_1/ReadVariableOp_3ReadVariableOp-lstm_1_readvariableop_lstm_1_recurrent_kernel^lstm_1/ReadVariableOp_2*
dtype0*
_output_shapes
:	 �m
lstm_1/strided_slice_6/stackConst*
valueB"    `   *
dtype0*
_output_shapes
:o
lstm_1/strided_slice_6/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:o
lstm_1/strided_slice_6/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
lstm_1/strided_slice_6StridedSlicelstm_1/ReadVariableOp_3:value:0%lstm_1/strided_slice_6/stack:output:0'lstm_1/strided_slice_6/stack_1:output:0'lstm_1/strided_slice_6/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:  �
lstm_1/MatMul_7MatMullstm_1/zeros:output:0lstm_1/strided_slice_6:output:0*'
_output_shapes
:��������� *
T0}
lstm_1/add_6AddV2lstm_1/BiasAdd_3:output:0lstm_1/MatMul_7:product:0*
T0*'
_output_shapes
:��������� S
lstm_1/Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: S
lstm_1/Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: p
lstm_1/Mul_4Mullstm_1/add_6:z:0lstm_1/Const_6:output:0*'
_output_shapes
:��������� *
T0p
lstm_1/Add_7Addlstm_1/Mul_4:z:0lstm_1/Const_7:output:0*
T0*'
_output_shapes
:��������� e
 lstm_1/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
valueB
 *  �?*
dtype0�
lstm_1/clip_by_value_2/MinimumMinimumlstm_1/Add_7:z:0)lstm_1/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:��������� ]
lstm_1/clip_by_value_2/yConst*
_output_shapes
: *
valueB
 *    *
dtype0�
lstm_1/clip_by_value_2Maximum"lstm_1/clip_by_value_2/Minimum:z:0!lstm_1/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:��������� Y
lstm_1/Tanh_1Tanhlstm_1/add_5:z:0*'
_output_shapes
:��������� *
T0t
lstm_1/mul_5Mullstm_1/clip_by_value_2:z:0lstm_1/Tanh_1:y:0*'
_output_shapes
:��������� *
T0u
$lstm_1/TensorArrayV2_1/element_shapeConst*
valueB"����    *
dtype0*
_output_shapes
:�
lstm_1/TensorArrayV2_1TensorListReserve-lstm_1/TensorArrayV2_1/element_shape:output:0lstm_1/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: M
lstm_1/timeConst*
value	B : *
dtype0*
_output_shapes
: j
lstm_1/while/maximum_iterationsConst*
valueB :
���������*
dtype0*
_output_shapes
: [
lstm_1/while/loop_counterConst*
dtype0*
_output_shapes
: *
value	B : �
lstm_1/whileWhile"lstm_1/while/loop_counter:output:0(lstm_1/while/maximum_iterations:output:0lstm_1/time:output:0lstm_1/TensorArrayV2_1:handle:0lstm_1/zeros:output:0lstm_1/zeros_1:output:0lstm_1/strided_slice_1:output:0>lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_1_split_readvariableop_lstm_1_kernel)lstm_1_split_1_readvariableop_lstm_1_bias-lstm_1_readvariableop_lstm_1_recurrent_kernel^lstm_1/ReadVariableOp_3^lstm_1/split/ReadVariableOp^lstm_1/split_1/ReadVariableOp*
_num_original_outputs*"
bodyR
lstm_1_while_body_6596*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *K
output_shapes:
8: : : : :��������� :��������� : : : : : *
T
2*
_lower_using_switch_merge(*
parallel_iterations *"
condR
lstm_1_while_cond_6595Y
lstm_1/while/IdentityIdentitylstm_1/while:output:0*
_output_shapes
: *
T0[
lstm_1/while/Identity_1Identitylstm_1/while:output:1*
T0*
_output_shapes
: [
lstm_1/while/Identity_2Identitylstm_1/while:output:2*
T0*
_output_shapes
: [
lstm_1/while/Identity_3Identitylstm_1/while:output:3*
T0*
_output_shapes
: l
lstm_1/while/Identity_4Identitylstm_1/while:output:4*'
_output_shapes
:��������� *
T0l
lstm_1/while/Identity_5Identitylstm_1/while:output:5*
T0*'
_output_shapes
:��������� [
lstm_1/while/Identity_6Identitylstm_1/while:output:6*
_output_shapes
: *
T0[
lstm_1/while/Identity_7Identitylstm_1/while:output:7*
T0*
_output_shapes
: [
lstm_1/while/Identity_8Identitylstm_1/while:output:8*
T0*
_output_shapes
: [
lstm_1/while/Identity_9Identitylstm_1/while:output:9*
_output_shapes
: *
T0]
lstm_1/while/Identity_10Identitylstm_1/while:output:10*
_output_shapes
: *
T0�
7lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
valueB"����    *
dtype0�
)lstm_1/TensorArrayV2Stack/TensorListStackTensorListStack lstm_1/while/Identity_3:output:0@lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:(��������� o
lstm_1/strided_slice_7/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:h
lstm_1/strided_slice_7/stack_1Const*
dtype0*
_output_shapes
:*
valueB: h
lstm_1/strided_slice_7/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
lstm_1/strided_slice_7StridedSlice2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_1/strided_slice_7/stack:output:0'lstm_1/strided_slice_7/stack_1:output:0'lstm_1/strided_slice_7/stack_2:output:0*'
_output_shapes
:��������� *
Index0*
T0*
shrink_axis_maskl
lstm_1/transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
lstm_1/transpose_1	Transpose2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������( �
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel*
dtype0*
_output_shapes

: �
dense/MatMulMatMullstm_1/strided_slice_7:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
dtype0*
_output_shapes
:�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:���������*
T0\

dense/ReluReludense/BiasAdd:output:0*'
_output_shapes
:���������*
T0Y
dropout/dropout/rateConst*
valueB
 *   ?*
dtype0*
_output_shapes
: ]
dropout/dropout/ShapeShapedense/Relu:activations:0*
_output_shapes
:*
T0g
"dropout/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: g
"dropout/dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  �?�
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:����������
"dropout/dropout/random_uniform/subSub+dropout/dropout/random_uniform/max:output:0+dropout/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
"dropout/dropout/random_uniform/mulMul5dropout/dropout/random_uniform/RandomUniform:output:0&dropout/dropout/random_uniform/sub:z:0*'
_output_shapes
:���������*
T0�
dropout/dropout/random_uniformAdd&dropout/dropout/random_uniform/mul:z:0+dropout/dropout/random_uniform/min:output:0*'
_output_shapes
:���������*
T0Z
dropout/dropout/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: z
dropout/dropout/subSubdropout/dropout/sub/x:output:0dropout/dropout/rate:output:0*
T0*
_output_shapes
: ^
dropout/dropout/truediv/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?�
dropout/dropout/truedivRealDiv"dropout/dropout/truediv/x:output:0dropout/dropout/sub:z:0*
T0*
_output_shapes
: �
dropout/dropout/GreaterEqualGreaterEqual"dropout/dropout/random_uniform:z:0dropout/dropout/rate:output:0*
T0*'
_output_shapes
:����������
dropout/dropout/mulMuldense/Relu:activations:0dropout/dropout/truediv:z:0*
T0*'
_output_shapes
:���������
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:����������
dropout/dropout/mul_1Muldropout/dropout/mul:z:0dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:����������
dense_1/MatMul/ReadVariableOpReadVariableOp,dense_1_matmul_readvariableop_dense_1_kernel*
dtype0*
_output_shapes

:�
dense_1/MatMulMatMuldropout/dropout/mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_1/BiasAdd/ReadVariableOpReadVariableOp+dense_1_biasadd_readvariableop_dense_1_bias*
dtype0*
_output_shapes
:�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentitydense_1/Sigmoid:y:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^features/embedding_lookup^lstm/ReadVariableOp^lstm/ReadVariableOp_1^lstm/ReadVariableOp_2^lstm/ReadVariableOp_3^lstm/split/ReadVariableOp^lstm/split_1/ReadVariableOp^lstm/while^lstm_1/ReadVariableOp^lstm_1/ReadVariableOp_1^lstm_1/ReadVariableOp_2^lstm_1/ReadVariableOp_3^lstm_1/split/ReadVariableOp^lstm_1/split_1/ReadVariableOp^lstm_1/while*'
_output_shapes
:���������*
T0"
identityIdentity:output:0*R
_input_shapesA
?:���������(:::::::::::2>
lstm_1/split_1/ReadVariableOplstm_1/split_1/ReadVariableOp26
lstm/split/ReadVariableOplstm/split/ReadVariableOp2

lstm/while
lstm/while2*
lstm/ReadVariableOplstm/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
lstm/split_1/ReadVariableOplstm/split_1/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp22
lstm_1/ReadVariableOp_1lstm_1/ReadVariableOp_122
lstm_1/ReadVariableOp_2lstm_1/ReadVariableOp_222
lstm_1/ReadVariableOp_3lstm_1/ReadVariableOp_32
lstm_1/whilelstm_1/while2.
lstm/ReadVariableOp_1lstm/ReadVariableOp_12.
lstm/ReadVariableOp_2lstm/ReadVariableOp_22.
lstm/ReadVariableOp_3lstm/ReadVariableOp_326
features/embedding_lookupfeatures/embedding_lookup2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2:
lstm_1/split/ReadVariableOplstm_1/split/ReadVariableOp2.
lstm_1/ReadVariableOplstm_1/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : 
��
�
>__inference_lstm_layer_call_and_return_conditional_losses_7679
inputs_0$
 split_readvariableop_lstm_kernel$
 split_1_readvariableop_lstm_bias(
$readvariableop_lstm_recurrent_kernel
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOp�while=
ShapeShapeinputs_0*
_output_shapes
:*
T0]
strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0_
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
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_maskM
zeros/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
_output_shapes
: *
T0O
zeros/Less/yConst*
dtype0*
_output_shapes
: *
value
B :�Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
_output_shapes
:*
T0*
NP
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@O
zeros_1/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@c
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
strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB: a
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
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
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
strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB: a
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
:���������*
T0*
Index0*
shrink_axis_maskG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
_output_shapes
: *
value	B :*
dtype0v
split/ReadVariableOpReadVariableOp split_readvariableop_lstm_kernel*
dtype0*
_output_shapes
:	��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split*<
_output_shapes*
(:@:@:@:@l
MatMulMatMulstrided_slice_2:output:0split:output:0*
T0*'
_output_shapes
:���������@n
MatMul_1MatMulstrided_slice_2:output:0split:output:1*
T0*'
_output_shapes
:���������@n
MatMul_2MatMulstrided_slice_2:output:0split:output:2*
T0*'
_output_shapes
:���������@n
MatMul_3MatMulstrided_slice_2:output:0split:output:3*
T0*'
_output_shapes
:���������@I
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: t
split_1/ReadVariableOpReadVariableOp split_1_readvariableop_lstm_bias*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split*,
_output_shapes
:@:@:@:@h
BiasAddBiasAddMatMul:product:0split_1:output:0*'
_output_shapes
:���������@*
T0l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:���������@l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:���������@l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:���������@t
ReadVariableOpReadVariableOp$readvariableop_lstm_recurrent_kernel*
dtype0*
_output_shapes
:	@�f
strided_slice_3/stackConst*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB"    @   h
strided_slice_3/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_3StridedSliceReadVariableOp:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:@@*
Index0*
T0n
MatMul_4MatMulzeros:output:0strided_slice_3:output:0*'
_output_shapes
:���������@*
T0d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:���������@L
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
: W
MulMuladd:z:0Const_2:output:0*'
_output_shapes
:���������@*
T0Y
Add_1AddMul:z:0Const_3:output:0*'
_output_shapes
:���������@*
T0\
clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������@T
clip_by_value/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������@�
ReadVariableOp_1ReadVariableOp$readvariableop_lstm_recurrent_kernel^ReadVariableOp*
dtype0*
_output_shapes
:	@�f
strided_slice_4/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_4/stack_1Const*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_4StridedSliceReadVariableOp_1:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:@@n
MatMul_5MatMulzeros:output:0strided_slice_4:output:0*
T0*'
_output_shapes
:���������@h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:���������@L
Const_4Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:���������@[
Add_3Add	Mul_1:z:0Const_5:output:0*'
_output_shapes
:���������@*
T0^
clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*'
_output_shapes
:���������@*
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
:���������@e
mul_2Mulclip_by_value_1:z:0zeros_1:output:0*'
_output_shapes
:���������@*
T0�
ReadVariableOp_2ReadVariableOp$readvariableop_lstm_recurrent_kernel^ReadVariableOp_1*
dtype0*
_output_shapes
:	@�f
strided_slice_5/stackConst*
_output_shapes
:*
valueB"    �   *
dtype0h
strided_slice_5/stack_1Const*
valueB"    �   *
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
end_mask*
_output_shapes

:@@*
Index0*
T0n
MatMul_6MatMulzeros:output:0strided_slice_5:output:0*
T0*'
_output_shapes
:���������@h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*'
_output_shapes
:���������@*
T0I
TanhTanh	add_4:z:0*'
_output_shapes
:���������@*
T0[
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:���������@V
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:���������@�
ReadVariableOp_3ReadVariableOp$readvariableop_lstm_recurrent_kernel^ReadVariableOp_2*
dtype0*
_output_shapes
:	@�f
strided_slice_6/stackConst*
_output_shapes
:*
valueB"    �   *
dtype0h
strided_slice_6/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_6/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_6StridedSliceReadVariableOp_3:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:@@*
Index0*
T0n
MatMul_7MatMulzeros:output:0strided_slice_6:output:0*
T0*'
_output_shapes
:���������@h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*'
_output_shapes
:���������@*
T0L
Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:���������@[
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:���������@^
clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:���������@V
clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*'
_output_shapes
:���������@*
T0K
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:���������@_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*'
_output_shapes
:���������@*
T0n
TensorArrayV2_1/element_shapeConst*
valueB"����@   *
dtype0*
_output_shapes
:�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
element_dtype0*
_output_shapes
: *

shape_type0F
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 split_readvariableop_lstm_kernel split_1_readvariableop_lstm_bias$readvariableop_lstm_recurrent_kernel^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_num_original_outputs*
bodyR
while_body_7526*L
_output_shapes:
8: : : : :���������@:���������@: : : : : *
T
2*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_7525K
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
:���������@^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:���������@M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
_output_shapes
: *
T0O
while/Identity_10Identitywhile:output:10*
_output_shapes
: *
T0�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"����@   *
dtype0*
_output_shapes
:�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :������������������@h
strided_slice_7/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:a
strided_slice_7/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_7StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*'
_output_shapes
:���������@*
T0*
Index0*
shrink_axis_maske
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������@�
IdentityIdentitytranspose_1:y:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp^while*4
_output_shapes"
 :������������������@*
T0"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::2
whilewhile2 
ReadVariableOpReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp:( $
"
_user_specified_name
inputs/0: : : 
�L
�
E__inference_lstm_cell_1_layer_call_and_return_conditional_losses_4153

inputs

states
states_1&
"split_readvariableop_lstm_1_kernel&
"split_1_readvariableop_lstm_1_bias*
&readvariableop_lstm_1_recurrent_kernel
identity

identity_1

identity_2��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOpG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: x
split/ReadVariableOpReadVariableOp"split_readvariableop_lstm_1_kernel*
dtype0*
_output_shapes
:	@��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
	num_split*<
_output_shapes*
(:@ :@ :@ :@ *
T0Z
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:��������� \
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:��������� \
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:��������� \
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:��������� I
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
dtype0*
_output_shapes
: *
value	B : v
split_1/ReadVariableOpReadVariableOp"split_1_readvariableop_lstm_1_bias*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split*,
_output_shapes
: : : : h
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:��������� l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:��������� l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:��������� l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*'
_output_shapes
:��������� *
T0v
ReadVariableOpReadVariableOp&readvariableop_lstm_1_recurrent_kernel*
dtype0*
_output_shapes
:	 �d
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
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:  d
MatMul_4MatMulstatesstrided_slice:output:0*'
_output_shapes
:��������� *
T0d
addAddV2BiasAdd:output:0MatMul_4:product:0*
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
: W
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:��������� Y
Add_1AddMul:z:0Const_3:output:0*
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
T0�
ReadVariableOp_1ReadVariableOp&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp*
_output_shapes
:	 �*
dtype0f
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

:  *
Index0*
T0f
MatMul_5MatMulstatesstrided_slice_1:output:0*'
_output_shapes
:��������� *
T0h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*'
_output_shapes
:��������� *
T0L
Const_4Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:��������� [
Add_3Add	Mul_1:z:0Const_5:output:0*
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
mul_2Mulclip_by_value_1:z:0states_1*
T0*'
_output_shapes
:��������� �
ReadVariableOp_2ReadVariableOp&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp_1*
dtype0*
_output_shapes
:	 �f
strided_slice_2/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_1Const*
valueB"    `   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:  f
MatMul_6MatMulstatesstrided_slice_2:output:0*'
_output_shapes
:��������� *
T0h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*'
_output_shapes
:��������� *
T0I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:��������� [
mul_3Mulclip_by_value:z:0Tanh:y:0*'
_output_shapes
:��������� *
T0V
add_5AddV2	mul_2:z:0	mul_3:z:0*'
_output_shapes
:��������� *
T0�
ReadVariableOp_3ReadVariableOp&readvariableop_lstm_1_recurrent_kernel^ReadVariableOp_2*
dtype0*
_output_shapes
:	 �f
strided_slice_3/stackConst*
valueB"    `   *
dtype0*
_output_shapes
:h
strided_slice_3/stack_1Const*
_output_shapes
:*
valueB"        *
dtype0h
strided_slice_3/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
end_mask*
_output_shapes

:  *
Index0*
T0*

begin_maskf
MatMul_7MatMulstatesstrided_slice_3:output:0*'
_output_shapes
:��������� *
T0h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*'
_output_shapes
:��������� *
T0L
Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:��������� [
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:��������� ^
clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*'
_output_shapes
:��������� *
T0V
clip_by_value_2/yConst*
dtype0*
_output_shapes
: *
valueB
 *    �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*'
_output_shapes
:��������� *
T0K
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:��������� _
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*'
_output_shapes
:��������� *
T0�
IdentityIdentity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:��������� �

Identity_1Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*'
_output_shapes
:��������� *
T0�

Identity_2Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:���������@:��������� :��������� :::2 
ReadVariableOpReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namestates:&"
 
_user_specified_namestates: : : 
�
�
lstm_while_cond_6320
lstm_while_loop_counter!
lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_lstm_strided_slice_10
,lstm_tensorarrayunstack_tensorlistfromtensor
lstm_kernel
	lstm_bias
lstm_recurrent_kernel
identity
U
LessLessplaceholderless_lstm_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
_output_shapes
: *
T0
"
identityIdentity:output:0*Q
_input_shapes@
>: : : : :���������@:���������@: : :::: : : : : : : : :	 :
 :  
�
�
(__inference_lstm_cell_layer_call_fn_9977

inputs
states_0
states_1'
#statefulpartitionedcall_lstm_kernel%
!statefulpartitionedcall_lstm_bias1
-statefulpartitionedcall_lstm_recurrent_kernel
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1#statefulpartitionedcall_lstm_kernel!statefulpartitionedcall_lstm_bias-statefulpartitionedcall_lstm_recurrent_kernel*+
_gradient_op_typePartitionedCall-3525*L
fGRE
C__inference_lstm_cell_layer_call_and_return_conditional_losses_3501*
Tout
2**
config_proto

GPU 

CPU2J 8*M
_output_shapes;
9:���������@:���������@:���������@*
Tin

2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*'
_output_shapes
:���������@*
T0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:���������:���������@:���������@:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0:($
"
_user_specified_name
states/1: : 
�
�
while_body_3939
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0)
%statefulpartitionedcall_lstm_kernel_0'
#statefulpartitionedcall_lstm_bias_03
/statefulpartitionedcall_lstm_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor'
#statefulpartitionedcall_lstm_kernel%
!statefulpartitionedcall_lstm_bias1
-statefulpartitionedcall_lstm_recurrent_kernel��StatefulPartitionedCall�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"����   �
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:����������
StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3%statefulpartitionedcall_lstm_kernel_0#statefulpartitionedcall_lstm_bias_0/statefulpartitionedcall_lstm_recurrent_kernel_0*+
_gradient_op_typePartitionedCall-3525*L
fGRE
C__inference_lstm_cell_layer_call_and_return_conditional_losses_3501*
Tout
2**
config_proto

GPU 

CPU2J 8*M
_output_shapes;
9:���������@:���������@:���������@*
Tin

2�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder StatefulPartitionedCall:output:0*
element_dtype0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
value	B :*
dtype0J
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: I
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
:���������@�

Identity_5Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@"L
#statefulpartitionedcall_lstm_kernel%statefulpartitionedcall_lstm_kernel_0"$
strided_slice_1strided_slice_1_0"`
-statefulpartitionedcall_lstm_recurrent_kernel/statefulpartitionedcall_lstm_recurrent_kernel_0"H
!statefulpartitionedcall_lstm_bias#statefulpartitionedcall_lstm_bias_0"!

identity_1Identity_1:output:0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0*Q
_input_shapes@
>: : : : :���������@:���������@: : :::22
StatefulPartitionedCallStatefulPartitionedCall:  : : : : : : : : :	 :
 
�
�
%__inference_lstm_1_layer_call_fn_9122
inputs_0)
%statefulpartitionedcall_lstm_1_kernel'
#statefulpartitionedcall_lstm_1_bias3
/statefulpartitionedcall_lstm_1_recurrent_kernel
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0%statefulpartitionedcall_lstm_1_kernel#statefulpartitionedcall_lstm_1_bias/statefulpartitionedcall_lstm_1_recurrent_kernel*+
_gradient_op_typePartitionedCall-4762*I
fDRB
@__inference_lstm_1_layer_call_and_return_conditional_losses_4761*
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
,:������������������@:::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0: : : 
�
�
while_cond_4547
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
lstm_1_kernel
lstm_1_bias
lstm_1_recurrent_kernel
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
identityIdentity:output:0*Q
_input_shapes@
>: : : : :��������� :��������� : : ::::  : : : : : : : : :	 :
 
�`
�
while_body_8379
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0&
"split_readvariableop_lstm_kernel_0&
"split_1_readvariableop_lstm_bias_0*
&readvariableop_lstm_recurrent_kernel_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor$
 split_readvariableop_lstm_kernel$
 split_1_readvariableop_lstm_bias(
$readvariableop_lstm_recurrent_kernel��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�split/ReadVariableOp�split_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"����   �
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:���������G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: x
split/ReadVariableOpReadVariableOp"split_readvariableop_lstm_kernel_0*
dtype0*
_output_shapes
:	��
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split*<
_output_shapes*
(:@:@:@:@~
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:0*'
_output_shapes
:���������@*
T0�
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:1*
T0*'
_output_shapes
:���������@�
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:2*
T0*'
_output_shapes
:���������@�
MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:3*'
_output_shapes
:���������@*
T0I
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
dtype0*
_output_shapes
: *
value	B : v
split_1/ReadVariableOpReadVariableOp"split_1_readvariableop_lstm_bias_0*
dtype0*
_output_shapes	
:��
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split*,
_output_shapes
:@:@:@:@h
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:���������@l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:���������@l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:���������@l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:���������@v
ReadVariableOpReadVariableOp&readvariableop_lstm_recurrent_kernel_0*
dtype0*
_output_shapes
:	@�d
strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:f
strided_slice/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0�
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:@@k
MatMul_4MatMulplaceholder_2strided_slice:output:0*
T0*'
_output_shapes
:���������@d
addAddV2BiasAdd:output:0MatMul_4:product:0*'
_output_shapes
:���������@*
T0L
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
 *   ?W
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:���������@Y
Add_1AddMul:z:0Const_3:output:0*
T0*'
_output_shapes
:���������@\
clip_by_value/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������@T
clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������@�
ReadVariableOp_1ReadVariableOp&readvariableop_lstm_recurrent_kernel_0^ReadVariableOp*
_output_shapes
:	@�*
dtype0f
strided_slice_2/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:h
strided_slice_2/stack_1Const*
valueB"    �   *
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

:@@*
Index0*
T0m
MatMul_5MatMulplaceholder_2strided_slice_2:output:0*
T0*'
_output_shapes
:���������@h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*'
_output_shapes
:���������@*
T0L
Const_4Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:���������@[
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*'
_output_shapes
:���������@^
clip_by_value_1/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*'
_output_shapes
:���������@*
T0V
clip_by_value_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*'
_output_shapes
:���������@*
T0b
mul_2Mulclip_by_value_1:z:0placeholder_3*
T0*'
_output_shapes
:���������@�
ReadVariableOp_2ReadVariableOp&readvariableop_lstm_recurrent_kernel_0^ReadVariableOp_1*
dtype0*
_output_shapes
:	@�f
strided_slice_3/stackConst*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_3/stack_1Const*
valueB"    �   *
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
end_mask*
_output_shapes

:@@m
MatMul_6MatMulplaceholder_2strided_slice_3:output:0*
T0*'
_output_shapes
:���������@h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*'
_output_shapes
:���������@*
T0I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:���������@[
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:���������@V
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:���������@�
ReadVariableOp_3ReadVariableOp&readvariableop_lstm_recurrent_kernel_0^ReadVariableOp_2*
dtype0*
_output_shapes
:	@�f
strided_slice_4/stackConst*
valueB"    �   *
dtype0*
_output_shapes
:h
strided_slice_4/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:h
strided_slice_4/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      �
strided_slice_4StridedSliceReadVariableOp_3:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*

begin_mask*
end_mask*
_output_shapes

:@@*
Index0*
T0m
MatMul_7MatMulplaceholder_2strided_slice_4:output:0*
T0*'
_output_shapes
:���������@h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*'
_output_shapes
:���������@*
T0L
Const_6Const*
valueB
 *��L>*
dtype0*
_output_shapes
: L
Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: [
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:���������@[
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:���������@^
clip_by_value_2/Minimum/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:���������@V
clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:���������@K
Tanh_1Tanh	add_5:z:0*'
_output_shapes
:���������@*
T0_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:���������@�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_5:z:0*
element_dtype0*
_output_shapes
: I
add_8/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_8AddV2placeholderadd_8/y:output:0*
T0*
_output_shapes
: I
add_9/yConst*
dtype0*
_output_shapes
: *
value	B :U
add_9AddV2while_loop_counteradd_9/y:output:0*
_output_shapes
: *
T0�
IdentityIdentity	add_9:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_1Identitywhile_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_2Identity	add_8:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_4Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:���������@�

Identity_5Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:���������@"F
 split_1_readvariableop_lstm_bias"split_1_readvariableop_lstm_bias_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"N
$readvariableop_lstm_recurrent_kernel&readvariableop_lstm_recurrent_kernel_0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"F
 split_readvariableop_lstm_kernel"split_readvariableop_lstm_kernel_0"$
strided_slice_1strided_slice_1_0*Q
_input_shapes@
>: : : : :���������@:���������@: : :::2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp2 
ReadVariableOpReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:  : : : : : : : : :	 :
 "�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*�
serving_default�
I
features_input7
 serving_default_features_input:0���������(;
dense_10
StatefulPartitionedCall:0���������tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:��
�=
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
	optimizer
	regularization_losses

trainable_variables
	variables
	keras_api

signatures
�__call__
+�&call_and_return_all_conditional_losses
�_default_save_signature"�9
_tf_keras_sequential�9{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": null, "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "Embedding", "config": {"name": "features", "trainable": true, "batch_input_shape": [null, 40], "dtype": "float32", "input_dim": 47506, "output_dim": 25, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null, "dtype": "float32"}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 40}}, {"class_name": "LSTM", "config": {"name": "lstm", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 1}}, {"class_name": "LSTM", "config": {"name": "lstm_1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 1}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 24, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": null, "activity_regularizer": null, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Embedding", "config": {"name": "features", "trainable": true, "batch_input_shape": [null, 40], "dtype": "float32", "input_dim": 47506, "output_dim": 25, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null, "dtype": "float32"}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 40}}, {"class_name": "LSTM", "config": {"name": "lstm", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 1}}, {"class_name": "LSTM", "config": {"name": "lstm_1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 1}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 24, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�
regularization_losses
trainable_variables
	variables
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "InputLayer", "name": "features_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 40], "config": {"batch_input_shape": [null, 40], "dtype": "float32", "sparse": false, "ragged": false, "name": "features_input"}, "input_spec": null, "activity_regularizer": null}
�

embeddings
_callable_losses
regularization_losses
trainable_variables
	variables
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Embedding", "name": "features", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 40], "config": {"name": "features", "trainable": true, "batch_input_shape": [null, 40], "dtype": "float32", "input_dim": 47506, "output_dim": 25, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null, "dtype": "float32"}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 40}, "input_spec": null, "activity_regularizer": null}
�
cell

state_spec
_callable_losses
regularization_losses
trainable_variables
	variables
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�	
_tf_keras_layer�	{"class_name": "LSTM", "name": "lstm", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "lstm", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 1}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 25], "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "activity_regularizer": null}
�
cell
 
state_spec
!_callable_losses
"regularization_losses
#trainable_variables
$	variables
%	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�	
_tf_keras_layer�	{"class_name": "LSTM", "name": "lstm_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "lstm_1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 1}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 64], "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "activity_regularizer": null}
�

&kernel
'bias
(_callable_losses
)regularization_losses
*trainable_variables
+	variables
,	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 24, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "activity_regularizer": null}
�
-_callable_losses
.regularization_losses
/trainable_variables
0	variables
1	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "input_spec": null, "activity_regularizer": null}
�

2kernel
3bias
4_callable_losses
5regularization_losses
6trainable_variables
7	variables
8	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 24}}}, "activity_regularizer": null}
�
9iter

:beta_1

;beta_2
	<decay
=learning_ratem�&m�'m�2m�3m�>m�?m�@m�Am�Bm�Cm�v�&v�'v�2v�3v�>v�?v�@v�Av�Bv�Cv�"
	optimizer
 "
trackable_list_wrapper
n
0
>1
?2
@3
A4
B5
C6
&7
'8
29
310"
trackable_list_wrapper
n
0
>1
?2
@3
A4
B5
C6
&7
'8
29
310"
trackable_list_wrapper
�
Dlayer_regularization_losses
	regularization_losses

Elayers
Fmetrics

trainable_variables
	variables
Gnon_trainable_variables
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Hlayer_regularization_losses
regularization_losses

Ilayers
Jmetrics
trainable_variables
	variables
Knon_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
':%
��2features/embeddings
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
�
Llayer_regularization_losses
regularization_losses

Mlayers
Nmetrics
trainable_variables
	variables
Onon_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�

>kernel
?recurrent_kernel
@bias
P_callable_losses
Qregularization_losses
Rtrainable_variables
S	variables
T	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "LSTMCell", "name": "lstm_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 64, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 1}, "input_spec": null, "activity_regularizer": null}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
>0
?1
@2"
trackable_list_wrapper
5
>0
?1
@2"
trackable_list_wrapper
�
Ulayer_regularization_losses
regularization_losses

Vlayers
Wmetrics
trainable_variables
	variables
Xnon_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�

Akernel
Brecurrent_kernel
Cbias
Y_callable_losses
Zregularization_losses
[trainable_variables
\	variables
]	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "LSTMCell", "name": "lstm_cell_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "lstm_cell_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 1}, "input_spec": null, "activity_regularizer": null}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
A0
B1
C2"
trackable_list_wrapper
5
A0
B1
C2"
trackable_list_wrapper
�
^layer_regularization_losses
"regularization_losses

_layers
`metrics
#trainable_variables
$	variables
anon_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
: 2dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
�
blayer_regularization_losses
)regularization_losses

clayers
dmetrics
*trainable_variables
+	variables
enon_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
flayer_regularization_losses
.regularization_losses

glayers
hmetrics
/trainable_variables
0	variables
inon_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :2dense_1/kernel
:2dense_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
�
jlayer_regularization_losses
5regularization_losses

klayers
lmetrics
6trainable_variables
7	variables
mnon_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2training/Adam/iter
: (2training/Adam/beta_1
: (2training/Adam/beta_2
: (2training/Adam/decay
%:# (2training/Adam/learning_rate
:	�2lstm/kernel
(:&	@�2lstm/recurrent_kernel
:�2	lstm/bias
 :	@�2lstm_1/kernel
*:(	 �2lstm_1/recurrent_kernel
:�2lstm_1/bias
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
'
n0"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
>0
?1
@2"
trackable_list_wrapper
5
>0
?1
@2"
trackable_list_wrapper
�
olayer_regularization_losses
Qregularization_losses

players
qmetrics
Rtrainable_variables
S	variables
rnon_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
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
A0
B1
C2"
trackable_list_wrapper
5
A0
B1
C2"
trackable_list_wrapper
�
slayer_regularization_losses
Zregularization_losses

tlayers
umetrics
[trainable_variables
\	variables
vnon_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
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
�
	wtotal
	xcount
y
_fn_kwargs
z_updates
{regularization_losses
|trainable_variables
}	variables
~	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MeanMetricWrapper", "name": "acc", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "acc", "dtype": "float32"}, "input_spec": null, "activity_regularizer": null}
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
w0
x1"
trackable_list_wrapper
�
layer_regularization_losses
{regularization_losses
�layers
�metrics
|trainable_variables
}	variables
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
5:3
��2#training/Adam/features/embeddings/m
,:* 2training/Adam/dense/kernel/m
&:$2training/Adam/dense/bias/m
.:,2training/Adam/dense_1/kernel/m
(:&2training/Adam/dense_1/bias/m
,:*	�2training/Adam/lstm/kernel/m
6:4	@�2%training/Adam/lstm/recurrent_kernel/m
&:$�2training/Adam/lstm/bias/m
.:,	@�2training/Adam/lstm_1/kernel/m
8:6	 �2'training/Adam/lstm_1/recurrent_kernel/m
(:&�2training/Adam/lstm_1/bias/m
5:3
��2#training/Adam/features/embeddings/v
,:* 2training/Adam/dense/kernel/v
&:$2training/Adam/dense/bias/v
.:,2training/Adam/dense_1/kernel/v
(:&2training/Adam/dense_1/bias/v
,:*	�2training/Adam/lstm/kernel/v
6:4	@�2%training/Adam/lstm/recurrent_kernel/v
&:$�2training/Adam/lstm/bias/v
.:,	@�2training/Adam/lstm_1/kernel/v
8:6	 �2'training/Adam/lstm_1/recurrent_kernel/v
(:&�2training/Adam/lstm_1/bias/v
�2�
)__inference_sequential_layer_call_fn_6169
)__inference_sequential_layer_call_fn_7369
)__inference_sequential_layer_call_fn_6128
)__inference_sequential_layer_call_fn_7385�
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
�2�
D__inference_sequential_layer_call_and_return_conditional_losses_6089
D__inference_sequential_layer_call_and_return_conditional_losses_6779
D__inference_sequential_layer_call_and_return_conditional_losses_7353
D__inference_sequential_layer_call_and_return_conditional_losses_6065�
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
__inference__wrapped_model_3273�
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
features_input���������(
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
'__inference_features_layer_call_fn_7400�
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
B__inference_features_layer_call_and_return_conditional_losses_7394�
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
#__inference_lstm_layer_call_fn_7966
#__inference_lstm_layer_call_fn_7974
#__inference_lstm_layer_call_fn_8540
#__inference_lstm_layer_call_fn_8548�
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
>__inference_lstm_layer_call_and_return_conditional_losses_8532
>__inference_lstm_layer_call_and_return_conditional_losses_7958
>__inference_lstm_layer_call_and_return_conditional_losses_8253
>__inference_lstm_layer_call_and_return_conditional_losses_7679�
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
�2�
%__inference_lstm_1_layer_call_fn_9122
%__inference_lstm_1_layer_call_fn_9696
%__inference_lstm_1_layer_call_fn_9114
%__inference_lstm_1_layer_call_fn_9688�
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
@__inference_lstm_1_layer_call_and_return_conditional_losses_9401
@__inference_lstm_1_layer_call_and_return_conditional_losses_9106
@__inference_lstm_1_layer_call_and_return_conditional_losses_8827
@__inference_lstm_1_layer_call_and_return_conditional_losses_9680�
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
$__inference_dense_layer_call_fn_9714�
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
?__inference_dense_layer_call_and_return_conditional_losses_9707�
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
&__inference_dropout_layer_call_fn_9749
&__inference_dropout_layer_call_fn_9744�
���
FullArgSpec)
args!�
jself
jinputs

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
A__inference_dropout_layer_call_and_return_conditional_losses_9734
A__inference_dropout_layer_call_and_return_conditional_losses_9739�
���
FullArgSpec)
args!�
jself
jinputs

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
�2�
&__inference_dense_1_layer_call_fn_9767�
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
A__inference_dense_1_layer_call_and_return_conditional_losses_9760�
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
"__inference_signature_wrapper_6187features_input
�2�
(__inference_lstm_cell_layer_call_fn_9963
(__inference_lstm_cell_layer_call_fn_9977�
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
C__inference_lstm_cell_layer_call_and_return_conditional_losses_9858
C__inference_lstm_cell_layer_call_and_return_conditional_losses_9949�
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
+__inference_lstm_cell_1_layer_call_fn_10187
+__inference_lstm_cell_1_layer_call_fn_10173�
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
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_10159
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_10068�
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
__inference__wrapped_model_3273y>@?ACB&'237�4
-�*
(�%
features_input���������(
� "1�.
,
dense_1!�
dense_1����������
#__inference_lstm_layer_call_fn_8540d>@??�<
5�2
$�!
inputs���������(

 
p

 
� "����������(@�
)__inference_sequential_layer_call_fn_7385`>@?ACB&'237�4
-�*
 �
inputs���������(
p 

 
� "�����������
#__inference_lstm_layer_call_fn_8548d>@??�<
5�2
$�!
inputs���������(

 
p 

 
� "����������(@�
D__inference_sequential_layer_call_and_return_conditional_losses_6779m>@?ACB&'237�4
-�*
 �
inputs���������(
p

 
� "%�"
�
0���������
� �
+__inference_lstm_cell_1_layer_call_fn_10173�ACB��}
v�s
 �
inputs���������@
K�H
"�
states/0��������� 
"�
states/1��������� 
p
� "c�`
�
0��������� 
A�>
�
1/0��������� 
�
1/1��������� �
)__inference_sequential_layer_call_fn_6169h>@?ACB&'23?�<
5�2
(�%
features_input���������(
p 

 
� "�����������
A__inference_dropout_layer_call_and_return_conditional_losses_9734\3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
%__inference_lstm_1_layer_call_fn_9688`ACB?�<
5�2
$�!
inputs���������(@

 
p

 
� "���������� �
B__inference_features_layer_call_and_return_conditional_losses_7394_/�,
%�"
 �
inputs���������(
� ")�&
�
0���������(
� �
D__inference_sequential_layer_call_and_return_conditional_losses_6065u>@?ACB&'23?�<
5�2
(�%
features_input���������(
p

 
� "%�"
�
0���������
� �
%__inference_lstm_1_layer_call_fn_9696`ACB?�<
5�2
$�!
inputs���������(@

 
p 

 
� "���������� �
>__inference_lstm_layer_call_and_return_conditional_losses_7679�>@?O�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "2�/
(�%
0������������������@
� �
>__inference_lstm_layer_call_and_return_conditional_losses_7958�>@?O�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "2�/
(�%
0������������������@
� �
D__inference_sequential_layer_call_and_return_conditional_losses_7353m>@?ACB&'237�4
-�*
 �
inputs���������(
p 

 
� "%�"
�
0���������
� �
@__inference_lstm_1_layer_call_and_return_conditional_losses_8827}ACBO�L
E�B
4�1
/�,
inputs/0������������������@

 
p

 
� "%�"
�
0��������� 
� �
A__inference_dropout_layer_call_and_return_conditional_losses_9739\3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
A__inference_dense_1_layer_call_and_return_conditional_losses_9760\23/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� �
+__inference_lstm_cell_1_layer_call_fn_10187�ACB��}
v�s
 �
inputs���������@
K�H
"�
states/0��������� 
"�
states/1��������� 
p 
� "c�`
�
0��������� 
A�>
�
1/0��������� 
�
1/1��������� �
@__inference_lstm_1_layer_call_and_return_conditional_losses_9106}ACBO�L
E�B
4�1
/�,
inputs/0������������������@

 
p 

 
� "%�"
�
0��������� 
� �
C__inference_lstm_cell_layer_call_and_return_conditional_losses_9858�>@?��}
v�s
 �
inputs���������
K�H
"�
states/0���������@
"�
states/1���������@
p
� "s�p
i�f
�
0/0���������@
E�B
�
0/1/0���������@
�
0/1/1���������@
� �
@__inference_lstm_1_layer_call_and_return_conditional_losses_9401mACB?�<
5�2
$�!
inputs���������(@

 
p

 
� "%�"
�
0��������� 
� �
>__inference_lstm_layer_call_and_return_conditional_losses_8532q>@??�<
5�2
$�!
inputs���������(

 
p 

 
� ")�&
�
0���������(@
� �
>__inference_lstm_layer_call_and_return_conditional_losses_8253q>@??�<
5�2
$�!
inputs���������(

 
p

 
� ")�&
�
0���������(@
� �
D__inference_sequential_layer_call_and_return_conditional_losses_6089u>@?ACB&'23?�<
5�2
(�%
features_input���������(
p 

 
� "%�"
�
0���������
� �
?__inference_dense_layer_call_and_return_conditional_losses_9707\&'/�,
%�"
 �
inputs��������� 
� "%�"
�
0���������
� �
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_10068�ACB��}
v�s
 �
inputs���������@
K�H
"�
states/0��������� 
"�
states/1��������� 
p
� "s�p
i�f
�
0/0��������� 
E�B
�
0/1/0��������� 
�
0/1/1��������� 
� y
&__inference_dropout_layer_call_fn_9744O3�0
)�&
 �
inputs���������
p
� "�����������
%__inference_lstm_1_layer_call_fn_9114pACBO�L
E�B
4�1
/�,
inputs/0������������������@

 
p

 
� "���������� �
C__inference_lstm_cell_layer_call_and_return_conditional_losses_9949�>@?��}
v�s
 �
inputs���������
K�H
"�
states/0���������@
"�
states/1���������@
p 
� "s�p
i�f
�
0/0���������@
E�B
�
0/1/0���������@
�
0/1/1���������@
� �
"__inference_signature_wrapper_6187�>@?ACB&'23I�F
� 
?�<
:
features_input(�%
features_input���������("1�.
,
dense_1!�
dense_1����������
(__inference_lstm_cell_layer_call_fn_9963�>@?��}
v�s
 �
inputs���������
K�H
"�
states/0���������@
"�
states/1���������@
p
� "c�`
�
0���������@
A�>
�
1/0���������@
�
1/1���������@y
&__inference_dropout_layer_call_fn_9749O3�0
)�&
 �
inputs���������
p 
� "�����������
%__inference_lstm_1_layer_call_fn_9122pACBO�L
E�B
4�1
/�,
inputs/0������������������@

 
p 

 
� "���������� �
@__inference_lstm_1_layer_call_and_return_conditional_losses_9680mACB?�<
5�2
$�!
inputs���������(@

 
p 

 
� "%�"
�
0��������� 
� }
'__inference_features_layer_call_fn_7400R/�,
%�"
 �
inputs���������(
� "����������(�
(__inference_lstm_cell_layer_call_fn_9977�>@?��}
v�s
 �
inputs���������
K�H
"�
states/0���������@
"�
states/1���������@
p 
� "c�`
�
0���������@
A�>
�
1/0���������@
�
1/1���������@y
&__inference_dense_1_layer_call_fn_9767O23/�,
%�"
 �
inputs���������
� "����������w
$__inference_dense_layer_call_fn_9714O&'/�,
%�"
 �
inputs��������� 
� "�����������
)__inference_sequential_layer_call_fn_6128h>@?ACB&'23?�<
5�2
(�%
features_input���������(
p

 
� "�����������
#__inference_lstm_layer_call_fn_7966}>@?O�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "%�"������������������@�
#__inference_lstm_layer_call_fn_7974}>@?O�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "%�"������������������@�
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_10159�ACB��}
v�s
 �
inputs���������@
K�H
"�
states/0��������� 
"�
states/1��������� 
p 
� "s�p
i�f
�
0/0��������� 
E�B
�
0/1/0��������� 
�
0/1/1��������� 
� �
)__inference_sequential_layer_call_fn_7369`>@?ACB&'237�4
-�*
 �
inputs���������(
p

 
� "����������