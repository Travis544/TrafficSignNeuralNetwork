Ст
оЊ
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
В
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Њ
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
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718йе
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
: *
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
: *
dtype0
В
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:  *
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
: *
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
†*А*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
†*А*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:А*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А+*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	А+*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:+*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:+*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
М
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d/kernel/m
Е
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
: *
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
: *
dtype0
Р
Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/conv2d_1/kernel/m
Й
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:  *
dtype0
А
Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
: *
dtype0
Д
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
†*А*$
shared_nameAdam/dense/kernel/m
}
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m* 
_output_shapes
:
†*А*
dtype0
{
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameAdam/dense/bias/m
t
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes	
:А*
dtype0
З
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А+*&
shared_nameAdam/dense_1/kernel/m
А
)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes
:	А+*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:+*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:+*
dtype0
М
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d/kernel/v
Е
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
: *
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
: *
dtype0
Р
Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/conv2d_1/kernel/v
Й
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:  *
dtype0
А
Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
: *
dtype0
Д
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
†*А*$
shared_nameAdam/dense/kernel/v
}
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v* 
_output_shapes
:
†*А*
dtype0
{
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameAdam/dense/bias/v
t
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes	
:А*
dtype0
З
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А+*&
shared_nameAdam/dense_1/kernel/v
А
)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes
:	А+*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:+*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:+*
dtype0

NoOpNoOp
“4
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Н4
valueГ4BА4 Bщ3
і
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
	optimizer
	regularization_losses

	variables
trainable_variables
	keras_api

signatures
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
 trainable_variables
!	keras_api
h

"kernel
#bias
$regularization_losses
%	variables
&trainable_variables
'	keras_api
R
(regularization_losses
)	variables
*trainable_variables
+	keras_api
h

,kernel
-bias
.regularization_losses
/	variables
0trainable_variables
1	keras_api
–
2iter

3beta_1

4beta_2
	5decay
6learning_ratemjmkmlmm"mn#mo,mp-mqvrvsvtvu"vv#vw,vx-vy
 
8
0
1
2
3
"4
#5
,6
-7
8
0
1
2
3
"4
#5
,6
-7
≠
	regularization_losses

7layers
8non_trainable_variables
9layer_metrics

	variables
:layer_regularization_losses
;metrics
trainable_variables
 
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
≠
regularization_losses

<layers
=non_trainable_variables
>layer_metrics
	variables
?layer_regularization_losses
@metrics
trainable_variables
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
≠
regularization_losses

Alayers
Bnon_trainable_variables
Clayer_metrics
	variables
Dlayer_regularization_losses
Emetrics
trainable_variables
 
 
 
≠
regularization_losses

Flayers
Gnon_trainable_variables
Hlayer_metrics
	variables
Ilayer_regularization_losses
Jmetrics
trainable_variables
 
 
 
≠
regularization_losses

Klayers
Lnon_trainable_variables
Mlayer_metrics
	variables
Nlayer_regularization_losses
Ometrics
 trainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

"0
#1

"0
#1
≠
$regularization_losses

Players
Qnon_trainable_variables
Rlayer_metrics
%	variables
Slayer_regularization_losses
Tmetrics
&trainable_variables
 
 
 
≠
(regularization_losses

Ulayers
Vnon_trainable_variables
Wlayer_metrics
)	variables
Xlayer_regularization_losses
Ymetrics
*trainable_variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

,0
-1

,0
-1
≠
.regularization_losses

Zlayers
[non_trainable_variables
\layer_metrics
/	variables
]layer_regularization_losses
^metrics
0trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
1
0
1
2
3
4
5
6
 
 
 

_0
`1
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
 
 
 
 
 
 
 
4
	atotal
	bcount
c	variables
d	keras_api
D
	etotal
	fcount
g
_fn_kwargs
h	variables
i	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

a0
b1

c	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

e0
f1

h	variables
|z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
П
serving_default_conv2d_inputPlaceholder*/
_output_shapes
:€€€€€€€€€*
dtype0*$
shape:€€€€€€€€€
Ї
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_inputconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€+**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference_signature_wrapper_12091
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
А
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *'
f"R 
__inference__traced_save_12452
з
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__traced_restore_12561ƒџ
£	
ћ
#__inference_signature_wrapper_12091
conv2d_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: 
	unknown_3:
†*А
	unknown_4:	А
	unknown_5:	А+
	unknown_6:+
identityИҐStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€+**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__wrapped_model_117262
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€+2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:€€€€€€€€€
&
_user_specified_nameconv2d_input
д
^
B__inference_flatten_layer_call_and_return_conditional_losses_11786

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€†*2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€†*2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€ :W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
д
^
B__inference_flatten_layer_call_and_return_conditional_losses_12263

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€†*2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€†*2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€ :W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
д5
Є
E__inference_sequential_layer_call_and_return_conditional_losses_12212

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource:  6
(conv2d_1_biasadd_readvariableop_resource: 8
$dense_matmul_readvariableop_resource:
†*А4
%dense_biasadd_readvariableop_resource:	А9
&dense_1_matmul_readvariableop_resource:	А+5
'dense_1_biasadd_readvariableop_resource:+
identityИҐconv2d/BiasAdd/ReadVariableOpҐconv2d/Conv2D/ReadVariableOpҐconv2d_1/BiasAdd/ReadVariableOpҐconv2d_1/Conv2D/ReadVariableOpҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOp™
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOpє
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
conv2d/Conv2D°
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp§
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d/Relu∞
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_1/Conv2D/ReadVariableOp“
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
conv2d_1/Conv2DІ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOpђ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_1/Relu√
max_pooling2d/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
flatten/ConstШ
flatten/ReshapeReshapemax_pooling2d/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€†*2
flatten/Reshape°
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
†*А*
dtype02
dense/MatMul/ReadVariableOpШ
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense/MatMulЯ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
dense/BiasAdd/ReadVariableOpЪ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

dense/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/ConstЮ
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/dropout/Mulv
dropout/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/ShapeЌ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЕ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/yя
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/dropout/GreaterEqualШ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€А2
dropout/dropout/CastЫ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/dropout/Mul_1¶
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	А+*
dtype02
dense_1/MatMul/ReadVariableOpЮ
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€+2
dense_1/MatMul§
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:+*
dtype02 
dense_1/BiasAdd/ReadVariableOp°
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€+2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€+2
dense_1/Softmaxн
IdentityIdentitydense_1/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€+2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€: : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ф
ь
C__inference_conv2d_1_layer_call_and_return_conditional_losses_11773

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
ш
т
E__inference_sequential_layer_call_and_return_conditional_losses_11830

inputs&
conv2d_11757: 
conv2d_11759: (
conv2d_1_11774:  
conv2d_1_11776: 
dense_11800:
†*А
dense_11802:	А 
dense_1_11824:	А+
dense_1_11826:+
identityИҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallП
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_11757conv2d_11759*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_117562 
conv2d/StatefulPartitionedCallЇ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_11774conv2d_1_11776*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_117732"
 conv2d_1/StatefulPartitionedCallН
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_117322
max_pooling2d/PartitionedCallс
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€†** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_117862
flatten/PartitionedCallЭ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_11800dense_11802*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_117992
dense/StatefulPartitionedCallс
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_118102
dropout/PartitionedCall¶
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_11824dense_1_11826*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_118232!
dense_1/StatefulPartitionedCallВ
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€+2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€: : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
≤!
Ъ
E__inference_sequential_layer_call_and_return_conditional_losses_12062
conv2d_input&
conv2d_12038: 
conv2d_12040: (
conv2d_1_12043:  
conv2d_1_12045: 
dense_12050:
†*А
dense_12052:	А 
dense_1_12056:	А+
dense_1_12058:+
identityИҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdropout/StatefulPartitionedCallХ
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_12038conv2d_12040*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_117562 
conv2d/StatefulPartitionedCallЇ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_12043conv2d_1_12045*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_117732"
 conv2d_1/StatefulPartitionedCallН
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_117322
max_pooling2d/PartitionedCallс
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€†** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_117862
flatten/PartitionedCallЭ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_12050dense_12052*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_117992
dense/StatefulPartitionedCallЙ
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_118792!
dropout/StatefulPartitionedCallЃ
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_12056dense_1_12058*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_118232!
dense_1/StatefulPartitionedCall§
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€+2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€: : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:] Y
/
_output_shapes
:€€€€€€€€€
&
_user_specified_nameconv2d_input
 
C
'__inference_flatten_layer_call_fn_12257

inputs
identityЅ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€†** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_117862
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€†*2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€ :W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
К 
ш
E__inference_sequential_layer_call_and_return_conditional_losses_12035
conv2d_input&
conv2d_12011: 
conv2d_12013: (
conv2d_1_12016:  
conv2d_1_12018: 
dense_12023:
†*А
dense_12025:	А 
dense_1_12029:	А+
dense_1_12031:+
identityИҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallХ
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_12011conv2d_12013*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_117562 
conv2d/StatefulPartitionedCallЇ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_12016conv2d_1_12018*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_117732"
 conv2d_1/StatefulPartitionedCallН
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_117322
max_pooling2d/PartitionedCallс
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€†** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_117862
flatten/PartitionedCallЭ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_12023dense_12025*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_117992
dense/StatefulPartitionedCallс
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_118102
dropout/PartitionedCall¶
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_12029dense_1_12031*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_118232!
dense_1/StatefulPartitionedCallВ
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€+2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€: : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:] Y
/
_output_shapes
:€€€€€€€€€
&
_user_specified_nameconv2d_input
у
`
B__inference_dropout_layer_call_and_return_conditional_losses_11810

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ѕ
Э
(__inference_conv2d_1_layer_call_fn_12241

inputs!
unknown:  
	unknown_0: 
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_117732
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
±
a
B__inference_dropout_layer_call_and_return_conditional_losses_12310

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yњ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Љ
C
'__inference_dropout_layer_call_fn_12288

inputs
identityЅ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_118102
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
э6
…
 __inference__wrapped_model_11726
conv2d_inputJ
0sequential_conv2d_conv2d_readvariableop_resource: ?
1sequential_conv2d_biasadd_readvariableop_resource: L
2sequential_conv2d_1_conv2d_readvariableop_resource:  A
3sequential_conv2d_1_biasadd_readvariableop_resource: C
/sequential_dense_matmul_readvariableop_resource:
†*А?
0sequential_dense_biasadd_readvariableop_resource:	АD
1sequential_dense_1_matmul_readvariableop_resource:	А+@
2sequential_dense_1_biasadd_readvariableop_resource:+
identityИҐ(sequential/conv2d/BiasAdd/ReadVariableOpҐ'sequential/conv2d/Conv2D/ReadVariableOpҐ*sequential/conv2d_1/BiasAdd/ReadVariableOpҐ)sequential/conv2d_1/Conv2D/ReadVariableOpҐ'sequential/dense/BiasAdd/ReadVariableOpҐ&sequential/dense/MatMul/ReadVariableOpҐ)sequential/dense_1/BiasAdd/ReadVariableOpҐ(sequential/dense_1/MatMul/ReadVariableOpЋ
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOpа
sequential/conv2d/Conv2DConv2Dconv2d_input/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
sequential/conv2d/Conv2D¬
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp–
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
sequential/conv2d/BiasAddЦ
sequential/conv2d/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
sequential/conv2d/Relu—
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOpю
sequential/conv2d_1/Conv2DConv2D$sequential/conv2d/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
sequential/conv2d_1/Conv2D»
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOpЎ
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
sequential/conv2d_1/BiasAddЬ
sequential/conv2d_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
sequential/conv2d_1/Reluд
 sequential/max_pooling2d/MaxPoolMaxPool&sequential/conv2d_1/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2"
 sequential/max_pooling2d/MaxPoolЕ
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
sequential/flatten/Constƒ
sequential/flatten/ReshapeReshape)sequential/max_pooling2d/MaxPool:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€†*2
sequential/flatten/Reshape¬
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
†*А*
dtype02(
&sequential/dense/MatMul/ReadVariableOpƒ
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential/dense/MatMulј
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp∆
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential/dense/BiasAddМ
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential/dense/ReluЮ
sequential/dropout/IdentityIdentity#sequential/dense/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential/dropout/Identity«
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes
:	А+*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp 
sequential/dense_1/MatMulMatMul$sequential/dropout/Identity:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€+2
sequential/dense_1/MatMul≈
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:+*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOpЌ
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€+2
sequential/dense_1/BiasAddЪ
sequential/dense_1/SoftmaxSoftmax#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€+2
sequential/dense_1/Softmax–
IdentityIdentity$sequential/dense_1/Softmax:softmax:0)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€+2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€: : : : : : : : 2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:] Y
/
_output_shapes
:€€€€€€€€€
&
_user_specified_nameconv2d_input
вН
б
!__inference__traced_restore_12561
file_prefix8
assignvariableop_conv2d_kernel: ,
assignvariableop_1_conv2d_bias: <
"assignvariableop_2_conv2d_1_kernel:  .
 assignvariableop_3_conv2d_1_bias: 3
assignvariableop_4_dense_kernel:
†*А,
assignvariableop_5_dense_bias:	А4
!assignvariableop_6_dense_1_kernel:	А+-
assignvariableop_7_dense_1_bias:+&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: B
(assignvariableop_17_adam_conv2d_kernel_m: 4
&assignvariableop_18_adam_conv2d_bias_m: D
*assignvariableop_19_adam_conv2d_1_kernel_m:  6
(assignvariableop_20_adam_conv2d_1_bias_m: ;
'assignvariableop_21_adam_dense_kernel_m:
†*А4
%assignvariableop_22_adam_dense_bias_m:	А<
)assignvariableop_23_adam_dense_1_kernel_m:	А+5
'assignvariableop_24_adam_dense_1_bias_m:+B
(assignvariableop_25_adam_conv2d_kernel_v: 4
&assignvariableop_26_adam_conv2d_bias_v: D
*assignvariableop_27_adam_conv2d_1_kernel_v:  6
(assignvariableop_28_adam_conv2d_1_bias_v: ;
'assignvariableop_29_adam_dense_kernel_v:
†*А4
%assignvariableop_30_adam_dense_bias_v:	А<
)assignvariableop_31_adam_dense_1_kernel_v:	А+5
'assignvariableop_32_adam_dense_1_bias_v:+
identity_34ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9ћ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*Ў
valueќBЋ"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names“
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЎ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ю
_output_shapesЛ
И::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЭ
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1£
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2І
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3•
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4§
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ґ
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¶
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7§
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8°
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9£
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10І
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¶
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ѓ
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13°
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14°
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15£
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16£
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17∞
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_conv2d_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ѓ
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_conv2d_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19≤
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_conv2d_1_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20∞
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_conv2d_1_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21ѓ
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_dense_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22≠
AssignVariableOp_22AssignVariableOp%assignvariableop_22_adam_dense_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23±
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24ѓ
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25∞
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_conv2d_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ѓ
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_conv2d_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27≤
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv2d_1_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28∞
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv2d_1_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29ѓ
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_dense_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30≠
AssignVariableOp_30AssignVariableOp%assignvariableop_30_adam_dense_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31±
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_1_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32ѓ
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_1_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_329
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpі
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_33І
Identity_34IdentityIdentity_33:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_34"#
identity_34Identity_34:output:0*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Т
ъ
A__inference_conv2d_layer_call_and_return_conditional_losses_11756

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
љ	
Ќ
*__inference_sequential_layer_call_fn_12112

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: 
	unknown_3:
†*А
	unknown_4:	А
	unknown_5:	А+
	unknown_6:+
identityИҐStatefulPartitionedCall√
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€+**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_118302
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€+2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ѕ	
”
*__inference_sequential_layer_call_fn_12008
conv2d_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: 
	unknown_3:
†*А
	unknown_4:	А
	unknown_5:	А+
	unknown_6:+
identityИҐStatefulPartitionedCall…
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€+**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_119682
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€+2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:€€€€€€€€€
&
_user_specified_nameconv2d_input
ґ

ф
B__inference_dense_1_layer_call_and_return_conditional_losses_11823

inputs1
matmul_readvariableop_resource:	А+-
biasadd_readvariableop_resource:+
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А+*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€+2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:+*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€+2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€+2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€+2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
і

ф
@__inference_dense_layer_call_and_return_conditional_losses_11799

inputs2
matmul_readvariableop_resource:
†*А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
†*А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€†*: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€†*
 
_user_specified_nameinputs
љ
Ы
&__inference_conv2d_layer_call_fn_12221

inputs!
unknown: 
	unknown_0: 
identityИҐStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_117562
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
®
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_11732

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ѕ	
”
*__inference_sequential_layer_call_fn_11849
conv2d_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: 
	unknown_3:
†*А
	unknown_4:	А
	unknown_5:	А+
	unknown_6:+
identityИҐStatefulPartitionedCall…
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€+**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_118302
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€+2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:€€€€€€€€€
&
_user_specified_nameconv2d_input
†!
Ф
E__inference_sequential_layer_call_and_return_conditional_losses_11968

inputs&
conv2d_11944: 
conv2d_11946: (
conv2d_1_11949:  
conv2d_1_11951: 
dense_11956:
†*А
dense_11958:	А 
dense_1_11962:	А+
dense_1_11964:+
identityИҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdropout/StatefulPartitionedCallП
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_11944conv2d_11946*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_117562 
conv2d/StatefulPartitionedCallЇ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_11949conv2d_1_11951*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_117732"
 conv2d_1/StatefulPartitionedCallН
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_117322
max_pooling2d/PartitionedCallс
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€†** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_117862
flatten/PartitionedCallЭ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_11956dense_11958*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_117992
dense/StatefulPartitionedCallЙ
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_118792!
dropout/StatefulPartitionedCallЃ
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_11962dense_1_11964*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_118232!
dense_1/StatefulPartitionedCall§
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€+2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€: : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Т
ъ
A__inference_conv2d_layer_call_and_return_conditional_losses_12232

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
—
I
-__inference_max_pooling2d_layer_call_fn_11738

inputs
identityй
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_117322
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
љ	
Ќ
*__inference_sequential_layer_call_fn_12133

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: 
	unknown_3:
†*А
	unknown_4:	А
	unknown_5:	А+
	unknown_6:+
identityИҐStatefulPartitionedCall√
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€+**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_119682
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€+2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
»
`
'__inference_dropout_layer_call_fn_12293

inputs
identityИҐStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_118792
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
в,
Є
E__inference_sequential_layer_call_and_return_conditional_losses_12169

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource:  6
(conv2d_1_biasadd_readvariableop_resource: 8
$dense_matmul_readvariableop_resource:
†*А4
%dense_biasadd_readvariableop_resource:	А9
&dense_1_matmul_readvariableop_resource:	А+5
'dense_1_biasadd_readvariableop_resource:+
identityИҐconv2d/BiasAdd/ReadVariableOpҐconv2d/Conv2D/ReadVariableOpҐconv2d_1/BiasAdd/ReadVariableOpҐconv2d_1/Conv2D/ReadVariableOpҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOp™
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOpє
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
conv2d/Conv2D°
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp§
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d/Relu∞
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_1/Conv2D/ReadVariableOp“
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
conv2d_1/Conv2DІ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOpђ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_1/Relu√
max_pooling2d/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
flatten/ConstШ
flatten/ReshapeReshapemax_pooling2d/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€†*2
flatten/Reshape°
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
†*А*
dtype02
dense/MatMul/ReadVariableOpШ
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense/MatMulЯ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
dense/BiasAdd/ReadVariableOpЪ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

dense/Relu}
dropout/IdentityIdentitydense/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/Identity¶
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	А+*
dtype02
dense_1/MatMul/ReadVariableOpЮ
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€+2
dense_1/MatMul§
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:+*
dtype02 
dense_1/BiasAdd/ReadVariableOp°
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€+2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€+2
dense_1/Softmaxн
IdentityIdentitydense_1/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€+2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€: : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ъ
Х
'__inference_dense_1_layer_call_fn_12319

inputs
unknown:	А+
	unknown_0:+
identityИҐStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_118232
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€+2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
чG
Ы
__inference__traced_save_12452
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename∆
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*Ў
valueќBЋ"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesћ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЗ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	2
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*І
_input_shapesХ
Т: : : :  : :
†*А:А:	А+:+: : : : : : : : : : : :  : :
†*А:А:	А+:+: : :  : :
†*А:А:	А+:+: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :&"
 
_output_shapes
:
†*А:!

_output_shapes	
:А:%!

_output_shapes
:	А+: 

_output_shapes
:+:	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :&"
 
_output_shapes
:
†*А:!

_output_shapes	
:А:%!

_output_shapes
:	А+: 

_output_shapes
:+:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :&"
 
_output_shapes
:
†*А:!

_output_shapes	
:А:% !

_output_shapes
:	А+: !

_output_shapes
:+:"

_output_shapes
: 
Ъ
Х
%__inference_dense_layer_call_fn_12272

inputs
unknown:
†*А
	unknown_0:	А
identityИҐStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_117992
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€†*: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€†*
 
_user_specified_nameinputs
Ф
ь
C__inference_conv2d_1_layer_call_and_return_conditional_losses_12252

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
±
a
B__inference_dropout_layer_call_and_return_conditional_losses_11879

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yњ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
у
`
B__inference_dropout_layer_call_and_return_conditional_losses_12298

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
і

ф
@__inference_dense_layer_call_and_return_conditional_losses_12283

inputs2
matmul_readvariableop_resource:
†*А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
†*А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€†*: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€†*
 
_user_specified_nameinputs
ґ

ф
B__inference_dense_1_layer_call_and_return_conditional_losses_12330

inputs1
matmul_readvariableop_resource:	А+-
biasadd_readvariableop_resource:+
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А+*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€+2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:+*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€+2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€+2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€+2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs"ћL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Љ
serving_default®
M
conv2d_input=
serving_default_conv2d_input:0€€€€€€€€€;
dense_10
StatefulPartitionedCall:0€€€€€€€€€+tensorflow/serving/predict:Ѕц
РA
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
	optimizer
	regularization_losses

	variables
trainable_variables
	keras_api

signatures
z__call__
{_default_save_signature
*|&call_and_return_all_conditional_losses"В>
_tf_keras_sequentialг={"name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 30, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 30, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 30, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 43, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 17}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 30, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 30, 30, 3]}, "float32", "conv2d_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 30, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}, "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 30, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 30, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 7}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 8}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 11}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 12}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 43, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 15}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 18}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Ћ

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
}__call__
*~&call_and_return_all_conditional_losses"¶

_tf_keras_layerМ
{"name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 30, 3]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 30, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 17}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 30, 3]}}
“

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
__call__
+А&call_and_return_all_conditional_losses"ђ

_tf_keras_layerТ
{"name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 30, 3]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 30, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 32]}}
ђ
regularization_losses
	variables
trainable_variables
	keras_api
Б__call__
+В&call_and_return_all_conditional_losses"Ы
_tf_keras_layerБ{"name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 20}}
У
regularization_losses
	variables
 trainable_variables
!	keras_api
Г__call__
+Д&call_and_return_all_conditional_losses"В
_tf_keras_layerи{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 21}}
“

"kernel
#bias
$regularization_losses
%	variables
&trainable_variables
'	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses"Ђ
_tf_keras_layerС{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5408}}, "shared_object_id": 22}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5408]}}
ы
(regularization_losses
)	variables
*trainable_variables
+	keras_api
З__call__
+И&call_and_return_all_conditional_losses"к
_tf_keras_layer–{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 12}
„

,kernel
-bias
.regularization_losses
/	variables
0trainable_variables
1	keras_api
Й__call__
+К&call_and_return_all_conditional_losses"∞
_tf_keras_layerЦ{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 43, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 23}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
г
2iter

3beta_1

4beta_2
	5decay
6learning_ratemjmkmlmm"mn#mo,mp-mqvrvsvtvu"vv#vw,vx-vy"
	optimizer
 "
trackable_list_wrapper
X
0
1
2
3
"4
#5
,6
-7"
trackable_list_wrapper
X
0
1
2
3
"4
#5
,6
-7"
trackable_list_wrapper
 
	regularization_losses

7layers
8non_trainable_variables
9layer_metrics

	variables
:layer_regularization_losses
;metrics
trainable_variables
z__call__
{_default_save_signature
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
-
Лserving_default"
signature_map
':% 2conv2d/kernel
: 2conv2d/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
≠
regularization_losses

<layers
=non_trainable_variables
>layer_metrics
	variables
?layer_regularization_losses
@metrics
trainable_variables
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
):'  2conv2d_1/kernel
: 2conv2d_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
ѓ
regularization_losses

Alayers
Bnon_trainable_variables
Clayer_metrics
	variables
Dlayer_regularization_losses
Emetrics
trainable_variables
__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞
regularization_losses

Flayers
Gnon_trainable_variables
Hlayer_metrics
	variables
Ilayer_regularization_losses
Jmetrics
trainable_variables
Б__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞
regularization_losses

Klayers
Lnon_trainable_variables
Mlayer_metrics
	variables
Nlayer_regularization_losses
Ometrics
 trainable_variables
Г__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
 :
†*А2dense/kernel
:А2
dense/bias
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
∞
$regularization_losses

Players
Qnon_trainable_variables
Rlayer_metrics
%	variables
Slayer_regularization_losses
Tmetrics
&trainable_variables
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞
(regularization_losses

Ulayers
Vnon_trainable_variables
Wlayer_metrics
)	variables
Xlayer_regularization_losses
Ymetrics
*trainable_variables
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
!:	А+2dense_1/kernel
:+2dense_1/bias
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
∞
.regularization_losses

Zlayers
[non_trainable_variables
\layer_metrics
/	variables
]layer_regularization_losses
^metrics
0trainable_variables
Й__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
‘
	atotal
	bcount
c	variables
d	keras_api"Э
_tf_keras_metricВ{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 24}
Ч
	etotal
	fcount
g
_fn_kwargs
h	variables
i	keras_api"–
_tf_keras_metricµ{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 18}
:  (2total
:  (2count
.
a0
b1"
trackable_list_wrapper
-
c	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
e0
f1"
trackable_list_wrapper
-
h	variables"
_generic_user_object
,:* 2Adam/conv2d/kernel/m
: 2Adam/conv2d/bias/m
.:,  2Adam/conv2d_1/kernel/m
 : 2Adam/conv2d_1/bias/m
%:#
†*А2Adam/dense/kernel/m
:А2Adam/dense/bias/m
&:$	А+2Adam/dense_1/kernel/m
:+2Adam/dense_1/bias/m
,:* 2Adam/conv2d/kernel/v
: 2Adam/conv2d/bias/v
.:,  2Adam/conv2d_1/kernel/v
 : 2Adam/conv2d_1/bias/v
%:#
†*А2Adam/dense/kernel/v
:А2Adam/dense/bias/v
&:$	А+2Adam/dense_1/kernel/v
:+2Adam/dense_1/bias/v
ц2у
*__inference_sequential_layer_call_fn_11849
*__inference_sequential_layer_call_fn_12112
*__inference_sequential_layer_call_fn_12133
*__inference_sequential_layer_call_fn_12008ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
л2и
 __inference__wrapped_model_11726√
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *3Ґ0
.К+
conv2d_input€€€€€€€€€
в2я
E__inference_sequential_layer_call_and_return_conditional_losses_12169
E__inference_sequential_layer_call_and_return_conditional_losses_12212
E__inference_sequential_layer_call_and_return_conditional_losses_12035
E__inference_sequential_layer_call_and_return_conditional_losses_12062ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
–2Ќ
&__inference_conv2d_layer_call_fn_12221Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_conv2d_layer_call_and_return_conditional_losses_12232Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_conv2d_1_layer_call_fn_12241Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_conv2d_1_layer_call_and_return_conditional_losses_12252Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Х2Т
-__inference_max_pooling2d_layer_call_fn_11738а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
∞2≠
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_11732а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
—2ќ
'__inference_flatten_layer_call_fn_12257Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_flatten_layer_call_and_return_conditional_losses_12263Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ѕ2ћ
%__inference_dense_layer_call_fn_12272Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
к2з
@__inference_dense_layer_call_and_return_conditional_losses_12283Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
М2Й
'__inference_dropout_layer_call_fn_12288
'__inference_dropout_layer_call_fn_12293і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
¬2њ
B__inference_dropout_layer_call_and_return_conditional_losses_12298
B__inference_dropout_layer_call_and_return_conditional_losses_12310і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
—2ќ
'__inference_dense_1_layer_call_fn_12319Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_dense_1_layer_call_and_return_conditional_losses_12330Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ѕBћ
#__inference_signature_wrapper_12091conv2d_input"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 †
 __inference__wrapped_model_11726|"#,-=Ґ:
3Ґ0
.К+
conv2d_input€€€€€€€€€
™ "1™.
,
dense_1!К
dense_1€€€€€€€€€+≥
C__inference_conv2d_1_layer_call_and_return_conditional_losses_12252l7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€ 
™ "-Ґ*
#К 
0€€€€€€€€€ 
Ъ Л
(__inference_conv2d_1_layer_call_fn_12241_7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€ 
™ " К€€€€€€€€€ ±
A__inference_conv2d_layer_call_and_return_conditional_losses_12232l7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "-Ґ*
#К 
0€€€€€€€€€ 
Ъ Й
&__inference_conv2d_layer_call_fn_12221_7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ " К€€€€€€€€€ £
B__inference_dense_1_layer_call_and_return_conditional_losses_12330],-0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "%Ґ"
К
0€€€€€€€€€+
Ъ {
'__inference_dense_1_layer_call_fn_12319P,-0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€+Ґ
@__inference_dense_layer_call_and_return_conditional_losses_12283^"#0Ґ-
&Ґ#
!К
inputs€€€€€€€€€†*
™ "&Ґ#
К
0€€€€€€€€€А
Ъ z
%__inference_dense_layer_call_fn_12272Q"#0Ґ-
&Ґ#
!К
inputs€€€€€€€€€†*
™ "К€€€€€€€€€А§
B__inference_dropout_layer_call_and_return_conditional_losses_12298^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ §
B__inference_dropout_layer_call_and_return_conditional_losses_12310^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "&Ґ#
К
0€€€€€€€€€А
Ъ |
'__inference_dropout_layer_call_fn_12288Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "К€€€€€€€€€А|
'__inference_dropout_layer_call_fn_12293Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "К€€€€€€€€€АІ
B__inference_flatten_layer_call_and_return_conditional_losses_12263a7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€ 
™ "&Ґ#
К
0€€€€€€€€€†*
Ъ 
'__inference_flatten_layer_call_fn_12257T7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€ 
™ "К€€€€€€€€€†*л
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_11732ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ √
-__inference_max_pooling2d_layer_call_fn_11738СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Ѕ
E__inference_sequential_layer_call_and_return_conditional_losses_12035x"#,-EҐB
;Ґ8
.К+
conv2d_input€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€+
Ъ Ѕ
E__inference_sequential_layer_call_and_return_conditional_losses_12062x"#,-EҐB
;Ґ8
.К+
conv2d_input€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€+
Ъ ї
E__inference_sequential_layer_call_and_return_conditional_losses_12169r"#,-?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€+
Ъ ї
E__inference_sequential_layer_call_and_return_conditional_losses_12212r"#,-?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€+
Ъ Щ
*__inference_sequential_layer_call_fn_11849k"#,-EҐB
;Ґ8
.К+
conv2d_input€€€€€€€€€
p 

 
™ "К€€€€€€€€€+Щ
*__inference_sequential_layer_call_fn_12008k"#,-EҐB
;Ґ8
.К+
conv2d_input€€€€€€€€€
p

 
™ "К€€€€€€€€€+У
*__inference_sequential_layer_call_fn_12112e"#,-?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p 

 
™ "К€€€€€€€€€+У
*__inference_sequential_layer_call_fn_12133e"#,-?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p

 
™ "К€€€€€€€€€+і
#__inference_signature_wrapper_12091М"#,-MҐJ
Ґ 
C™@
>
conv2d_input.К+
conv2d_input€€€€€€€€€"1™.
,
dense_1!К
dense_1€€€€€€€€€+