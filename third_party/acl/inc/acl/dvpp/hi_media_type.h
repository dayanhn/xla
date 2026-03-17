/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef HI_MEDIA_TYPE_H_
#define HI_MEDIA_TYPE_H_
#include <stdint.h>

typedef unsigned char hi_u8;
typedef signed char hi_s8;
typedef unsigned short hi_u16;
typedef short hi_s16;
typedef unsigned int hi_u32;
typedef int hi_s32;
typedef unsigned long long hi_u64;
typedef long long hi_s64;
typedef char hi_char;
typedef double hi_double;
typedef hi_u32 hi_fr32;
typedef float hi_float;

#define hi_void void
#define HI_NULL 0L
#define HI_SUCCESS 0
#define HI_FAILURE (-1)

typedef enum {
    HI_FALSE = 0,
    HI_TRUE = 1,
} hi_bool;

#endif // #ifndef HI_MEDIA_TYPE_H_
