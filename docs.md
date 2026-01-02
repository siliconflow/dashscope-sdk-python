需要以generational顶尖计算机理论科学家的严谨，清洗和整理设计文档
注意要明确写清楚要支持哪些HTTP API接口，输入输出格式，错误码定义等细节。



deepseek三方伙伴接入规格

1. 1模型协议要求

1. 百炼基于http/sse的协议去调伙伴网关，至少需要实现dashscope协议的输入输出格式&错误码格式，保证sse每包都透出usage信息，然后百炼网关再按需转换成对外的openai协议

2. dashscope协议具体的输入输出格式、各字段含义在文档中有完整定义https://help.aliyun.com/zh/model-studio/qwen-api-reference#a9b7b197e2q2v

3. 错误码对齐文档中的定义 https://help.aliyun.com/zh/model-studio/error-code

4. 流式返回的usage字段中需要包括input_tokens/output_tokens/total_tokens的完整信息



Dashscope API 提交任务接口调用

（POST https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation）



功能描述：

采用的是 http 同步接口来完成客户的响应，目前我们提供普通 http 和 http sse 两种协议，客户可以根据自己的需求和喜好自行选择。



入参描述（message版）：

字段

类型

传参方式

必选

描述

示例值

Content-Type

String

Header

是

请求类型：application/json 或者text/event-stream（开启 SSE 响应）

application/json

Authorization

String

Header

是

API-Key，例如：Bearer d1**2a

Bearer d1**2a

X-DashScope-SSE

String

Header

否

跟Accept: text/event-stream 二选一即可启用SSE响应

enable

model

String

Body

是

指明需要调用的模型

deepseek-v3

input.messages

List

Body

是

用户与模型的对话历史。list中的每个元素形式为{"role":角色, "content": 内容}。role=user为必传参数

"messages":[

            {

                "role": "system",

                "content": "You are a helpful assistant."

            },

            {

                "role": "user",

                "content": "你是谁？"

            }

        ]

input.messages.role

String

Body

是

角色当前可选值：system、user、assistant、tool和plugin。其中，仅messages[0]中支持role为system，普通chat模式下user和assistant需要交替出现，其它场景（包括function、plugin、file和多模态等）无此限制。

input.messages.content

String

Body

是

消息内容

parameters.temperature

Float

Body

否

采样温度，控制模型生成文本的多样性。temperature越高，生成的文本更多样，反之，生成的文本更确定。取值范围： [0, 2)

1.0

parameters.top_p

Float

Body

否

生成时，核采样方法的概率阈值。例如，取值为0.8时，仅保留累计概率之和大于等于0.8的概率分布中的token，作为随机采样的候选集。取值范围为(0,1.0)，取值越大，生成的随机性越高；取值越低，生成的随机性越低。默认值 0.8。注意，取值不要大于等于1。

0.8

parameters.top_k

Integer

Body

 否

生成时，采样候选集的大小。例如，取值为50时，仅将单次生成中得分最高的50个token组成随机采样的候选集。取值越大，生成的随机性越高；取值越小，生成的确定性越高。注意：如果top_k的值大于100，top_k将采用默认值100。

50

parameters.seed

Integer

Body

否

生成时，随机数的种子，用于控制模型生成的随机性。如果使用相同的种子，每次运行生成的结果都将相同；当需要复现模型的生成结果时，可以使用相同的种子。seed参数支持无符号64位整数类型。默认值 1234。

56789

parameters.max_tokens

Integer

Body

否

用于限制模型输出的最大 Token 数。若生成内容超过此值，生成将提前停止，且返回的finish_reason为length。适用于需控制输出长度的场景，如生成摘要、关键词，或用于降低成本、缩短响应时间。触发 max_tokens 时，响应的 finish_reason 字段为 length。

500

parameters.enable_search

Bool

Body

否

生成时，是否参考夸克搜索的结果。注意：打开搜索并不意味着一定会使用搜索结果；如果打开搜索，模型会将搜索结果作为prompt，进而“自行判断”是否生成结合搜索结果的文本。

true 或者 false

parameters.enable_thinking

    Bool

    Body

  否

生成时是否进行思考。enable_thinking 为 false 时不进行思考，enable_thinking 为 true 时由模型自主决定是否思考。默认 enable_thinking 为 False，注意：当设置 enable_thinking 为 true时， API 的某些行为可能不再工作：

1. 只支持增量输出 incremental_output=true

2. 不支持续写

3. 支持吃 result_format='message'

4. 不支持 http 访问 （请确认）

true 或者 false

parameters.thinking_budget

Integer

Body

否

思考过程的最大长度。

1024

parameters.incremental_output

Boolean

Body

否

在流式输出模式下是否开启增量输出。

false

parameters.result_format

String

Body

否

返回数据的格式。deepseek模型仅支持返回message格式

message

parameters.logprobs

Boolean

Body

否

是否返回输出 Token 的对数概率

fasle

parameters.top_logprobs

Integer

Body

否

指定在每一步生成时，返回模型最大概率的候选 Token 个数。取值范围：[0,5]

1

parameters.n

Integer

Body

否

生成响应的个数，取值范围是1-4。对于需要生成多个响应的场景（如创意写作、广告文案等），可以设置较大的 n 值。

1

parameters.stop

String

Body

否

用于指定停止词。当模型生成的文本中出现stop 指定的字符串或token_id时，生成将立即终止。可传入敏感词以控制模型的输出。

hi

入参描述（prompt版）：

字段

类型

传参方式

必选

描述

示例值

Content-Type

String

Header

是

请求类型：application/json 或者text/event-stream（开启 SSE 响应）

application/json

Authorization

String

Header

是

API-Key，例如：Bearer d1**2a

Bearer d1**2a

X-DashScope-SSE

String

Header

否

跟Accept: text/event-stream 二选一即可启用SSE响应

enable

model

String

Body

是

指明需要调用的模型，目前可选 qwen-v1 和 qwen-plus-v1。

qwen-v1

input.prompt

String

Body

否

用户当前输入的期望模型执行指令，支持中英文；qwen-v1 prompt字段支持 1.5k Tokens 长度，qwen-plus-v1 prompt字段支持 6.5k Tokens 长度。

哪个公园距离我更近

input.history

List

Body

否

用户与模型的对话历史，list中的每个元素是形式为{"user":"用户输入","bot":"模型输出"}的一轮对话，多轮对话按时间正序排列。

"history": [

    {

        "user":"今天天气好吗？",

        "bot":"今天天气不错，要出去玩玩嘛？"

    },

    {

        "user":"那你有什么地方推荐？",

        "bot":"我建议你去公园，春天来了，花朵开了，很美丽。"

    }

]

input.history.user

String

Body

否

用户输入

input.history.bot

String

Body

否

模型输出

parameters







同message协议版本



出参描述（message 版）：

字段

类型

描述

示例值

output.text

String

本次请求的算法输出内容。

null

output.choices[]

List





          output.choices[0].finish_reason

      String

有三种情况：正在生成时为null，生成结束时如果由于停止token导致则为stop，生成结束时如果因为生成长度过长导致则为length。

stop

output.choices[0].message.role

 String

回复角色

assistant

output.choices[0].message.content

String

回复内容

我建议你去颐和园

output.choices[0].message.reasoning_content

String

思考内容

用户让我推荐北京的景点....

total_tokens

Integer

当输入为纯文本时返回该字段，为input_tokens与output_tokens之和。

39

usage.input_tokens

Integer

本次请求输入内容的 token 数目。在打开了搜索的情况下，输入的 token 数目因为还需要添加搜索相关内容支持，所以会超出客户在请求中的输入。

22

usage.output_tokens

Integer

本次请求算法输出内容的 token 数目。

17

usage.output_tokens_details.text_tokens

Integer

输出的文本转换为Token后的长度。



usage.output_tokens_details.reasoning_tokens

Integer

思考过程转换为Token后的长度。



request_id

String

本次请求的系统唯一码

7574ee8f-38a3-4b1e-9280-11c33ab46e51

Dashscope协议输入示例

curl --location "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation" \

--header "Authorization: Bearer $DASHSCOPE_API_KEY" \

--header "Content-Type: application/json" \

--header "X-DashScope-SSE: enable" \

--data '{

    "model": "deepseek-r1",

    "input":{

        "messages":[

            {

                "role": "user",

                "content": "你是谁？"

            }

        ]

    },

    "parameters": {

        "result_format": "message",

        "incremental_output":true,

        "max_tokens": 1024

    }

}'

{

    "output": {

        "choices": [

            {

                "message": {

                    "content": "",

                    "reasoning_content": "嗯",

                    "role": "assistant"

                },

                "finish_reason": "null"

            }

        ]

    },

    "usage": {

        "total_tokens": 8,

        "output_tokens": 3,

        "input_tokens": 5,

        "output_tokens_details": {

            "reasoning_tokens": 1

        }

    },

    "request_id": "5354a793-a130-4b36-b1dc-2610c5f49071"

}



2. 模型计量计费要求

对于用户cancel（首包前cancel、首包后cancel）、绿网未通过时、触发tpm限流时等情况时的计量计费逻辑比较复杂，需要以Dashscope根据usage字段计量的最终数据为准，以此进行推账、bi分析等操作。

3. 其他：

tool call请参考百炼对外文档



通Qwen，非thinking时候支持auto\required\none\还有json。thinking时候支持auto和none



Function Calling

https://help.aliyun.com/zh/model-studio/qwen-function-calling?spm=a2c4g.11186623.help-menu-2400256.d_0_2_8_1.7c235e66rNzmo4&scm=20140722.H_2862208._.OR_help-T_cn~zh-V_1#cec840ca39b8b







注意：

1、该协议只针对deepseek有效，不缺分版本（v3、v3.1、v3.2、r1通用）

2、message版和prompt版的区别是什么？

· prompt是老协议，message新协议，但是都需要支持协议的兼容。

· 出参统一都是message版本

3、dashscope协议具体的输入输出格式、各字段都需要按照文档进行实现，以确保用户的使用

4、错误码需要与dashscope进行映射，确保三方模型厂商的错误码是dashscope的子集

5、流式返回的usage每一个包都需要包含：input_tokens/output_tokens/total_tokens







 通义千问API参考更新时间：2025-12-30 19:49:05

产品详情



我的收藏

本文介绍通义千问 API 的输入与输出参数，并提供 Python 等主流语言在典型场景下的调用示例。

模型介绍、选型建议和使用方法请参考文本生成模型概述。

可通过 OpenAI 兼容或 DashScope 协议调用通义千问 API。

OpenAI 兼容

北京地域新加坡地域金融云





SDK 调用配置的base_url：https://dashscope.aliyuncs.com/compatible-mode/v1

HTTP 请求地址：POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions

您需要先获取与配置 API Key并配置API Key到环境变量。若通过OpenAI SDK进行调用，需要安装SDK。

请求体

POST

/chat/completions



调试

文本输入流式输出图像输入视频输入工具调用联网搜索异步调用文档理解





PythonJavaNode.jsGoC#（HTTP）PHP（HTTP）curl







import osfrom openai import OpenAI



client = OpenAI(

    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"

    api_key=os.getenv("DASHSCOPE_API_KEY"),

    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",)



completion = client.chat.completions.create(

    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models

    model="qwen-plus",

    messages=[

        {"role": "system", "content": "You are a helpful assistant."},

        {"role": "user", "content": "你是谁？"},

    ])print(completion.model_dump_json())

model string （必选）

模型名称。

支持的模型：Qwen 大语言模型（商业版、开源版）、Qwen-VL、Qwen-Coder、Qwen-Omni、Qwen-Math。

Qwen-Audio不支持OpenAI兼容协议，仅支持DashScope协议。

具体模型名称和计费，请参见模型列表。

messages array （必选）

传递给大模型的上下文，按对话顺序排列。

消息类型



System Message object （可选）

系统消息，用于设定大模型的角色、语气、任务目标或约束条件等。一般放在messages数组的第一位。

QwQ 模型不建议设置 System Message，QVQ 模型设置 System Message不会生效。

属性



User Message object （必选）

用户消息，用于向模型传递问题、指令或上下文等。

属性



Assistant Message object （可选）

模型的回复。通常用于在多轮对话中作为上下文回传给模型。

属性



Tool Message object （可选）

工具的输出信息。

属性



stream boolean （可选） 默认值为 false

是否以流式输出方式回复。相关文档：流式输出

可选值：

false：模型生成全部内容后一次性返回；

true：边生成边输出，每生成一部分内容即返回一个数据块（chunk）。需实时逐个读取这些块以拼接完整回复。

推荐设置为true，可提升阅读体验并降低超时风险。

stream_options object （可选）

流式输出的配置项，仅在 stream 为 true 时生效。

属性



include_usage boolean （可选）默认值为false

是否在响应的最后一个数据块包含Token消耗信息。

可选值：

true：包含；

false：不包含。

流式输出时，Token 消耗信息仅可出现在响应的最后一个数据块。

modalities array （可选）默认值为["text"]

输出数据的模态，仅适用于 Qwen-Omni 模型。相关文档：全模态

可选值：

["text","audio"]：输出文本与音频；

["text"]：仅输出文本。

audio object （可选）

输出音频的音色与格式，仅适用于 Qwen-Omni 模型，且modalities参数需为["text","audio"]。相关文档：全模态

属性



temperature float （可选）

采样温度，控制模型生成文本的多样性。

temperature越高，生成的文本更多样，反之，生成的文本更确定。

取值范围： [0, 2)

temperature与top_p均可以控制生成文本的多样性，建议只设置其中一个值。更多说明，请参见文本生成模型概述。

temperature默认值



不建议修改QVQ模型的默认temperature值 。

top_p float （可选）

核采样的概率阈值，控制模型生成文本的多样性。

top_p越高，生成的文本更多样。反之，生成的文本更确定。

取值范围：（0,1.0]

temperature与top_p均可以控制生成文本的多样性，建议只设置其中一个值。更多说明，请参见文本生成模型概述。

top_p默认值



不建议修改QVQ模型的默认 top_p 值。

top_k integer （可选）

指定生成过程中用于采样的候选 Token 数量。值越大，输出越随机；值越小，输出越确定。若设为 null 或大于 100，则禁用 top_k 策略，仅 top_p 策略生效。取值必须为大于或等于 0 的整数。

top_k默认值



该参数非OpenAI标准参数。通过 Python SDK调用时，请放入 extra_body 对象中。配置方式为：extra_body={"top_k":xxx}。

不建议修改QVQ模型的默认 top_k 值。

presence_penalty float （可选）

控制模型生成文本时的内容重复度。

取值范围：[-2.0, 2.0]。正值降低重复度，负值增加重复度。

在创意写作或头脑风暴等需要多样性、趣味性或创造力的场景中，建议调高该值；在技术文档或正式文本等强调一致性与术语准确性的场景中，建议调低该值。

presence_penalty默认值



原理介绍



示例



使用qwen-vl-plus-2025-01-25模型进行文字提取时，建议设置presence_penalty为1.5。

不建议修改QVQ模型的默认presence_penalty值。

response_format object （可选） 默认值为{"type": "text"}

返回内容的格式。可选值：

{"type": "text"}：输出文字回复；

{"type": "json_object"}：输出标准格式的JSON字符串。

{"type": "json_schema","json_schema": {...} }：输出指定格式的JSON字符串。

相关文档：结构化输出。

若指定为{"type": "json_object"}，需在提示词中明确指示模型输出JSON，如：“请按照json格式输出”，否则会报错。

支持的模型参见结构化输出。

属性



max_input_tokens integer （可选）

允许输入的最大 Token 长度。目前仅支持qwen-plus-0728/latest模型。

qwen-plus-latest 默认值：129,024

后续默认值可能调整至1,000,000。

qwen-plus-2025-07-28 默认值：1,000,000

该参数非OpenAI标准参数。通过 Python SDK调用时，请放入 extra_body 对象中。配置方式为：extra_body={"max_input_tokens": xxx}。

max_tokens integer （可选）

用于限制模型输出的最大 Token 数。若生成内容超过此值，生成将提前停止，且返回的finish_reason为length。

默认值与最大值均为模型的最大输出长度，请参见模型列表。

适用于需控制输出长度的场景，如生成摘要、关键词，或用于降低成本、缩短响应时间。

触发 max_tokens 时，响应的 finish_reason 字段为 length。

max_tokens不限制思考模型思维链的长度。

vl_high_resolution_images boolean （可选）默认值为false

是否将输入图像的像素上限提升至 16384 Token 对应的像素值。相关文档：处理高分辨率图像。

vl_high_resolution_images：true，使用固定分辨率策略，忽略 max_pixels 设置，超过此分辨率时会将图像总像素缩小至此上限内。

点击查看各模型像素上限



vl_high_resolution_images为false，实际分辨率由 max_pixels 与默认上限共同决定，取二者计算结果的最大值。超过此像素上限时会将图像缩小至此上限内。

点击查看各模型的默认像素上限



该参数非OpenAI标准参数。通过 Python SDK调用时，请放入 extra_body 对象中。配置方式为：extra_body={"vl_high_resolution_images":xxx}。

n integer （可选） 默认值为1

生成响应的数量，取值范围是1-4。适用于需生成多个候选响应的场景，例如创意写作或广告文案。

仅支持 qwen-plus、 Qwen3（非思考模式）、qwen-plus-character 模型。

若传入 tools 参数， 请将n 设为 1。

增大 n 会增加输出 Token 的消耗，但不增加输入 Token 消耗。

enable_thinking boolean （可选）

使用混合思考（回复前既可思考也可不思考）模型时，是否开启思考模式。适用于 Qwen3 、Qwen3-Omni-Flash、Qwen3-VL模型。相关文档：深度思考

可选值：

true：开启

开启后，思考内容将通过reasoning_content字段返回。

false：不开启

不同模型的默认值：支持的模型

通该参数非OpenAI标准参数。通过 Python SDK调用时，请放入 extra_body 对象中。配置方式为：extra_body={"enable_thinking": xxx}。

thinking_budget integer （可选）

思考过程的最大 Token 数。适用于Qwen3-VL、Qwen3 的商业版与开源版模型。相关文档：限制思考长度。

默认值为模型最大思维链长度，请参见：模型列表

该参数非OpenAI标准参数。通过 Python SDK调用时，请放入 extra_body 对象中。配置方式为：extra_body={"thinking_budget": xxx}。

enable_code_interpreter boolean （可选）默认值为 false

是否开启代码解释器功能。仅当model为qwen3-max-preview且enable_thinking为true时生效。相关文档：代码解释器

可选值：

true：开启

false：不开启

该参数非OpenAI标准参数。通过 Python SDK调用时，请放入 extra_body 对象中。配置方式为：extra_body={"enable_code_interpreter": xxx}。

seed integer （可选）

随机数种子。用于确保在相同输入和参数下生成结果可复现。若调用时传入相同的 seed 且其他参数不变，模型将尽可能返回相同结果。

取值范围：[0,231−1]。

seed默认值



logprobs boolean （可选）默认值为 false

是否返回输出 Token 的对数概率，可选值：

true

返回

false

不返回

思考阶段生成的内容（reasoning_content）不会返回对数概率。

支持的模型



top_logprobs integer （可选）默认值为0

指定在每一步生成时，返回模型最大概率的候选 Token 个数。

取值范围：[0,5]

仅当 logprobs 为 true 时生效。

stop string 或 array （可选）

用于指定停止词。当模型生成的文本中出现stop 指定的字符串或token_id时，生成将立即终止。

可传入敏感词以控制模型的输出。

stop为数组时，不可将token_id和字符串同时作为元素输入，比如不可以指定为["你好",104307]。

tools array （可选）

包含一个或多个工具对象的数组，供模型在 Function Calling 中调用。相关文档：Function Calling

设置 tools 且模型判断需要调用工具时，响应会通过 tool_calls 返回工具信息。

属性



tool_choice string 或 object （可选）默认值为 auto

工具选择策略。若需对某类问题强制指定工具调用方式（例如始终使用某工具或禁用所有工具），可设置此参数。

可选值：

auto

大模型自主选择工具策略。

none

若不希望进行工具调用，可设定tool_choice参数为none；

{"type": "function", "function": {"name": "the_function_to_call"}}

若希望强制调用某个工具，可设定tool_choice参数为{"type": "function", "function": {"name": "the_function_to_call"}}，其中the_function_to_call是指定的工具函数名称。

思考模式的模型不支持强制调用某个工具。

parallel_tool_calls boolean （可选）默认值为 false

是否开启并行工具调用。相关文档：并行工具调用

可选值：

true：开启

false：不开启

enable_search boolean （可选）默认值为 false

是否开启联网搜索。相关文档：联网搜索

可选值：

true：开启；

若开启后未联网搜索，可优化提示词，或设置search_options中的forced_search参数开启强制搜索。

false：不开启。

启用互联网搜索功能可能会增加 Token 的消耗。

该参数非OpenAI标准参数。通过 Python SDK调用时，请放入 extra_body 对象中。配置方式为：extra_body={"enable_search": True}。

search_options object （可选）

联网搜索的策略。相关文档：联网搜索

属性



该参数非OpenAI标准参数。通过 Python SDK调用时，请放入 extra_body 对象中。配置方式为：extra_body={"search_options": xxx}。

X-DashScope-DataInspection string （可选）

在通义千问 API 的内容安全能力基础上，是否进一步识别输入输出内容的违规信息。取值如下：

'{"input":"cip","output":"cip"}'：进一步识别；

不设置该参数：不进一步识别。

通过 HTTP 调用时请放入请求头：-H "X-DashScope-DataInspection: {\"input\": \"cip\", \"output\": \"cip\"}"；

通过 Python SDK 调用时请通过extra_headers配置：extra_headers={'X-DashScope-DataInspection': '{"input":"cip","output":"cip"}'}。

详细使用方法请参见内容审核。

不支持通过 Node.js SDK设置。

chat响应对象（非流式输出）



{

    "choices": [

        {

            "message": {

                "role": "assistant",

                "content": "我是阿里云开发的一款超大规模语言模型，我叫通义千问。"

            },

            "finish_reason": "stop",

            "index": 0,

            "logprobs": null

        }

    ],

    "object": "chat.completion",

    "usage": {

        "prompt_tokens": 3019,

        "completion_tokens": 104,

        "total_tokens": 3123,

        "prompt_tokens_details": {

            "cached_tokens": 2048

        }

    },

    "created": 1735120033,

    "system_fingerprint": null,

    "model": "qwen-plus",

    "id": "chatcmpl-6ada9ed2-7f33-9de2-8bb0-78bd4035025a"

}

id string

本次调用的唯一标识符。

choices array

模型生成内容的数组。

属性



created integer

请求创建时的 Unix 时间戳（秒）。

model string

本次请求使用的模型。

object string

始终为chat.completion。

service_tier string

该参数当前固定为null。

system_fingerprint string

该参数当前固定为null。

usage object

本次请求的 Token 消耗信息。

属性



chat响应chunk对象（流式输出）



{"id":"chatcmpl-e30f5ae7-3063-93c4-90fe-beb5f900bd57","choices":[{"delta":{"content":"","function_call":null,"refusal":null,"role":"assistant","tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1735113344,"model":"qwen-plus","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}

{"id":"chatcmpl-e30f5ae7-3063-93c4-90fe-beb5f900bd57","choices":[{"delta":{"content":"我是","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1735113344,"model":"qwen-plus","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}

{"id":"chatcmpl-e30f5ae7-3063-93c4-90fe-beb5f900bd57","choices":[{"delta":{"content":"来自","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1735113344,"model":"qwen-plus","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}

{"id":"chatcmpl-e30f5ae7-3063-93c4-90fe-beb5f900bd57","choices":[{"delta":{"content":"阿里","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1735113344,"model":"qwen-plus","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}

{"id":"chatcmpl-e30f5ae7-3063-93c4-90fe-beb5f900bd57","choices":[{"delta":{"content":"云的超大规模","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1735113344,"model":"qwen-plus","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}

{"id":"chatcmpl-e30f5ae7-3063-93c4-90fe-beb5f900bd57","choices":[{"delta":{"content":"语言模型，我","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1735113344,"model":"qwen-plus","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}

{"id":"chatcmpl-e30f5ae7-3063-93c4-90fe-beb5f900bd57","choices":[{"delta":{"content":"叫通义千","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1735113344,"model":"qwen-plus","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}

{"id":"chatcmpl-e30f5ae7-3063-93c4-90fe-beb5f900bd57","choices":[{"delta":{"content":"问。","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1735113344,"model":"qwen-plus","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}

{"id":"chatcmpl-e30f5ae7-3063-93c4-90fe-beb5f900bd57","choices":[{"delta":{"content":"","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":"stop","index":0,"logprobs":null}],"created":1735113344,"model":"qwen-plus","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}

{"id":"chatcmpl-e30f5ae7-3063-93c4-90fe-beb5f900bd57","choices":[],"created":1735113344,"model":"qwen-plus","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":{"completion_tokens":17,"prompt_tokens":22,"total_tokens":39,"completion_tokens_details":null,"prompt_tokens_details":{"audio_tokens":null,"cached_tokens":0}}}

id string

本次调用的唯一标识符。每个chunk对象有相同的 id。

choices array

模型生成内容的数组，可包含一个或多个对象。若设置include_usage参数为true，则choices在最后一个chunk中为空数组。

属性



created integer

本次请求被创建时的时间戳。每个chunk有相同的时间戳。

model string

本次请求使用的模型。

object string

始终为chat.completion.chunk。

service_tier string

该参数当前固定为null。

system_fingerprintstring

该参数当前固定为null。

usage object

本次请求消耗的Token。只在include_usage为true时，在最后一个chunk显示。

属性



DashScope

北京地域新加坡地域金融云





HTTP 请求地址：

通义千问大语言模型：POST https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation

通义千问VL/Audio模型：POST https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation

SDK 调用无需配置 base_url。

您需要已获取与配置 API Key并配置API Key到环境变量。如果通过DashScope SDK进行调用，需要安装DashScope SDK。

请求体

文本输入流式输出图像输入视频输入音频输入联网搜索工具调用异步调用文档理解





PythonJavaPHP（HTTP）Node.js（HTTP）C#（HTTP）Go（HTTP）curl







import osimport dashscope



messages = [

    {'role': 'system', 'content': 'You are a helpful assistant.'},

    {'role': 'user', 'content': '你是谁？'}]

response = dashscope.Generation.call(

    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"

    api_key=os.getenv('DASHSCOPE_API_KEY'),

    model="qwen-plus", # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models

    messages=messages,

    result_format='message'

    )print(response)

model string （必选）

模型名称。

支持的模型：Qwen 大语言模型（商业版、开源版）、Qwen-VL、Qwen-Coder、通义千问Audio、数学模型。

具体模型名称和计费，请参见模型列表。

messages array （必选）

传递给大模型的上下文，按对话顺序排列。

通过HTTP调用时，请将messages 放入 input 对象中。

消息类型



temperature float （可选）

采样温度，控制模型生成文本的多样性。

temperature越高，生成的文本更多样，反之，生成的文本更确定。

取值范围： [0, 2)

temperature默认值



通过HTTP调用时，请将 temperature 放入 parameters 对象中。

不建议修改QVQ模型的默认 temperature 值。

top_p float （可选）

核采样的概率阈值，控制模型生成文本的多样性。

top_p越高，生成的文本更多样。反之，生成的文本更确定。

取值范围：（0,1.0]。

top_p默认值



Java SDK中为topP。通过HTTP调用时，请将 top_p 放入 parameters 对象中。

不建议修改QVQ模型的默认 top_p 值。

top_k integer （可选）

生成过程中采样候选集的大小。例如，取值为50时，仅将单次生成中得分最高的50个Token组成随机采样的候选集。取值越大，生成的随机性越高；取值越小，生成的确定性越高。取值为None或当top_k大于100时，表示不启用top_k策略，此时仅有top_p策略生效。

取值需要大于或等于0。

top_k默认值



Java SDK中为topK。通过HTTP调用时，请将 top_k 放入 parameters 对象中。

不建议修改QVQ模型的默认 top_k 值。

enable_thinking boolean （可选）

使用混合思考模型时，是否开启思考模式，适用于 Qwen3 、Qwen3-VL模型。相关文档：深度思考

可选值：

true：开启

开启后，思考内容将通过reasoning_content字段返回。

false：不开启

不同模型的默认值：支持的模型

Java SDK 为enableThinking；通过HTTP调用时，请将 enable_thinking 放入 parameters 对象中。

thinking_budget integer （可选）

思考过程的最大长度。适用于Qwen3-VL、Qwen3 的商业版与开源版模型。相关文档：限制思考长度。

默认值为模型最大思维链长度，请参见：模型列表

Java SDK 为 thinkingBudget。通过HTTP调用时，请将 thinking_budget 放入 parameters 对象中。

默认值为模型最大思维链长度。

enable_code_interpreter boolean （可选）默认值为 false

是否开启代码解释器功能。仅适用于思考模式下的 qwen3-max-preview。相关文档：代码解释器

可选值：

true：开启

false：不开启

不支持 Java SDK。通过HTTP调用时，请将 enable_code_interpreter 放入 parameters 对象中。

repetition_penalty float （可选）

模型生成时连续序列中的重复度。提高repetition_penalty时可以降低模型生成的重复度，1.0表示不做惩罚。没有严格的取值范围，只要大于0即可。

repetition_penalty默认值



Java SDK中为repetitionPenalty。通过HTTP调用时，请将 repetition_penalty 放入 parameters 对象中。

使用qwen-vl-plus_2025-01-25模型进行文字提取时，建议设置repetition_penalty为1.0。

不建议修改QVQ模型的默认 repetition_penalty 值。

presence_penalty float （可选）

控制模型生成文本时的内容重复度。

取值范围：[-2.0, 2.0]。正值降低重复度，负值增加重复度。

在创意写作或头脑风暴等需要多样性、趣味性或创造力的场景中，建议调高该值；在技术文档或正式文本等强调一致性与术语准确性的场景中，建议调低该值。

presence_penalty默认值



原理介绍



示例



使用qwen-vl-plus-2025-01-25模型进行文字提取时，建议设置presence_penalty为1.5。

不建议修改QVQ模型的默认presence_penalty值。

Java SDK不支持设置该参数。通过HTTP调用时，请将 presence_penalty 放入 parameters 对象中。

vl_high_resolution_images boolean （可选）默认值为false

是否将输入图像的像素上限提升至 16384 Token 对应的像素值。相关文档：处理高分辨率图像。

vl_high_resolution_images：true，使用固定分辨率策略，忽略 max_pixels 设置，超过此分辨率时会将图像总像素缩小至此上限内。

点击查看各模型像素上限



vl_high_resolution_images为false，实际分辨率由 max_pixels 与默认上限共同决定，取二者计算结果的最大值。超过此像素上限时会将图像缩小至此上限内。

点击查看各模型的默认像素上限



Java SDK 为 vlHighResolutionImages（需要的最低版本为2.20.8）。通过HTTP调用时，请将 vl_high_resolution_images 放入 parameters 对象中。

vl_enable_image_hw_output boolean （可选）默认值为 false

是否返回图像缩放后的尺寸。模型会对输入的图像进行缩放处理，配置为 True 时会返回图像缩放后的高度和宽度，开启流式输出时，该信息在最后一个数据块（chunk）中返回。支持Qwen-VL模型。

Java SDK中为 vlEnableImageHwOutput，Java SDK最低版本为2.20.8。通过HTTP调用时，请将 vl_enable_image_hw_output 放入 parameters 对象中。

max_input_tokens integer （可选）

允许输入的最大 Token 长度。目前仅支持qwen-plus-0728/latest模型。

qwen-plus-latest 默认值：129,024

后续默认值可能调整至1,000,000。

qwen-plus-2025-07-28 默认值：1,000,000

Java SDK 暂不支持该参数。通过HTTP调用时，请将 max_input_tokens 放入 parameters 对象中。

max_tokens integer （可选）

用于限制模型输出的最大 Token 数。若生成内容超过此值，生成将提前停止，且返回的finish_reason为length。

默认值与最大值均为模型的最大输出长度，请参见模型列表。

适用于需控制输出长度的场景，如生成摘要、关键词，或用于降低成本、缩短响应时间。

触发 max_tokens 时，响应的 finish_reason 字段为 length。

max_tokens不限制思考模型思维链的长度。

Java SDK中为maxTokens（模型为通义千问VL/Audio时，Java SDK中为maxLength，在 2.18.4 版本之后支持也设置为 maxTokens）。通过HTTP调用时，请将 max_tokens 放入 parameters 对象中。

seed integer （可选）

随机数种子。用于确保在相同输入和参数下生成结果可复现。若调用时传入相同的 seed 且其他参数不变，模型将尽可能返回相同结果。

取值范围：[0,231−1]。

seed默认值



通过HTTP调用时，请将 seed 放入 parameters 对象中。

stream boolean （可选） 默认值为false

是否流式输出回复。参数值：

false：模型生成完所有内容后一次性返回结果。

true：边生成边输出，即每生成一部分内容就立即输出一个片段（chunk）。

该参数仅支持Python SDK。通过Java SDK实现流式输出请通过streamCall接口调用；通过HTTP实现流式输出请在Header中指定X-DashScope-SSE为enable。

Qwen3商业版（思考模式）、Qwen3开源版、QwQ、QVQ只支持流式输出。

incremental_output boolean （可选）默认为false（Qwen3-Max、Qwen3-VL、Qwen3 开源版、QwQ 、QVQ模型默认值为 true）

在流式输出模式下是否开启增量输出。推荐您优先设置为true。

参数值：

false：每次输出为当前已经生成的整个序列，最后一次输出为生成的完整结果。





I

I like

I like apple

I like apple.

true（推荐）：增量输出，即后续输出内容不包含已输出的内容。您需要实时地逐个读取这些片段以获得完整的结果。





I

like

apple

.

Java SDK中为incrementalOutput。通过HTTP调用时，请将 incremental_output 放入 parameters 对象中。

QwQ 模型与思考模式下的 Qwen3 模型只支持设置为 true。由于 Qwen3 商业版模型默认值为false，您需要在思考模式下手动设置为 true。

Qwen3 开源版模型不支持设置为 false。

response_format object （可选） 默认值为{"type": "text"}

返回内容的格式。可选值：

{"type": "text"}：输出文字回复；

{"type": "json_object"}：输出标准格式的JSON字符串。

{"type": "json_schema","json_schema": {...} }：输出指定格式的JSON字符串。

相关文档：结构化输出。

支持的模型参见支持的模型。

若指定为{"type": "json_object"}，需在提示词中明确指示模型输出JSON，如：“请按照json格式输出”，否则会报错。

Java SDK中为responseFormat。通过HTTP调用时，请将 response_format 放入 parameters 对象中。

属性



result_format string （可选） 默认为text（Qwen3-Max、Qwen3-VL、QwQ 模型、Qwen3 开源模型（除了qwen3-next-80b-a3b-instruct）与 Qwen-Long 模型默认值为 message）

返回数据的格式。推荐您优先设置为message，可以更方便地进行多轮对话。

平台后续将统一调整默认值为message。

Java SDK中为resultFormat。通过HTTP调用时，请将 result_format 放入 parameters 对象中。

模型为通义千问VL/QVQ/Audio时，设置text不生效。

Qwen3-Max、Qwen3-VL、思考模式下的 Qwen3 模型只能设置为message，由于 Qwen3 商业版模型默认值为text，您需要将其设置为message。

如果您使用 Java SDK 调用Qwen3 开源模型，并且传入了 text，依然会以 message格式进行返回。

logprobs boolean （可选）默认值为 false

是否返回输出 Token 的对数概率，可选值：

true

返回

false

不返回

支持以下模型：

qwen-plus系列的快照模型（不包含主线模型）

qwen-turbo 系列的快照模型（不包含主线模型）

Qwen3 开源模型

通过HTTP调用时，请将 logprobs 放入 parameters 对象中。

top_logprobs integer （可选）默认值为0

指定在每一步生成时，返回模型最大概率的候选 Token 个数。

取值范围：[0,5]

仅当 logprobs 为 true 时生效。

Java SDK中为topLogprobs。通过HTTP调用时，请将 top_logprobs 放入 parameters 对象中。

n integer （可选） 默认值为1

生成响应的个数，取值范围是1-4。对于需要生成多个响应的场景（如创意写作、广告文案等），可以设置较大的 n 值。

当前仅支持 qwen-plus、 Qwen3（非思考模式）、qwen-plus-character 模型，且在传入 tools 参数时固定为1。

设置较大的 n 值不会增加输入 Token 消耗，会增加输出 Token 的消耗。

通过HTTP调用时，请将 n 放入 parameters 对象中。

stop string 或 array （可选）

用于指定停止词。当模型生成的文本中出现stop 指定的字符串或token_id时，生成将立即终止。

可传入敏感词以控制模型的输出。

stop为数组时，不可将token_id和字符串同时作为元素输入，比如不可以指定为["你好",104307]。

通过HTTP调用时，请将 stop 放入 parameters 对象中。

tools array （可选）

包含一个或多个工具对象的数组，供模型在 Function Calling 中调用。相关文档：Function Calling

使用 tools 时，必须将result_format设为message。

发起 Function Calling，或提交工具执行结果时，都必须设置tools参数。

属性



通过HTTP调用时，请将 tools 放入 parameters 对象中。暂时不支持qwen-vl与qwen-audio系列模型。

tool_choice string 或 object （可选）默认值为 auto

工具选择策略。若需对某类问题强制指定工具调用方式（例如始终使用某工具或禁用所有工具），可设置此参数。

auto

大模型自主选择工具策略；

none

若在特定请求中希望临时禁用工具调用，可设定tool_choice参数为none；

{"type": "function", "function": {"name": "the_function_to_call"}}

若希望强制调用某个工具，可设定tool_choice参数为{"type": "function", "function": {"name": "the_function_to_call"}}，其中the_function_to_call是指定的工具函数名称。

思考模式的模型不支持强制调用某个工具。

Java SDK中为toolChoice。通过HTTP调用时，请将 tool_choice 放入 parameters 对象中。

parallel_tool_calls boolean （可选）默认值为 false

是否开启并行工具调用。

可选值：

true：开启

false：不开启。

并行工具调用详情请参见：并行工具调用。

Java SDK中为parallelToolCalls。通过HTTP调用时，请将 parallel_tool_calls 放入 parameters 对象中。

enable_search boolean （可选） 默认值为false

模型在生成文本时是否使用互联网搜索结果进行参考。取值如下：

true：启用互联网搜索，模型会将搜索结果作为文本生成过程中的参考信息，但模型会基于其内部逻辑判断是否使用互联网搜索结果。

若开启后未联网搜索，可优化提示词，或设置search_options中的forced_search参数开启强制搜索。

false：关闭互联网搜索。

计费信息请参见计费说明。

Java SDK中为enableSearch。通过HTTP调用时，请将 enable_search 放入 parameters 对象中。

启用互联网搜索功能可能会增加 Token 的消耗。

search_options object （可选）

联网搜索的策略。仅当enable_search为true时生效。详情参见联网搜索。

通过HTTP调用时，请将 search_options 放入 parameters 对象中。Java SDK中为searchOptions。

属性



X-DashScope-DataInspection string （可选）

在通义千问 API 的内容安全能力基础上，是否进一步识别输入输出内容的违规信息。取值如下：

'{"input":"cip","output":"cip"}'：进一步识别；

不设置该参数：不进一步识别。

通过 HTTP 调用时请放入请求头：-H "X-DashScope-DataInspection: {\"input\": \"cip\", \"output\": \"cip\"}"；

通过 Python SDK 调用时请通过headers配置：headers={'X-DashScope-DataInspection': '{"input":"cip","output":"cip"}'}。

详细使用方法请参见内容审核。

不支持通过 Java SDK 设置。

不适用于Qwen-Audio 系列模型。

chat响应对象（流式与非流式输出格式一致）



{

  "status_code": 200,

  "request_id": "902fee3b-f7f0-9a8c-96a1-6b4ea25af114",

  "code": "",

  "message": "",

  "output": {

    "text": null,

    "finish_reason": null,

    "choices": [

      {

        "finish_reason": "stop",

        "message": {

          "role": "assistant",

          "content": "我是阿里云开发的一款超大规模语言模型，我叫通义千问。"

        }

      }

    ]

  },

  "usage": {

    "input_tokens": 22,

    "output_tokens": 17,

    "total_tokens": 39

  }

}

status_code string

本次请求的状态码。200 表示请求成功，否则表示请求失败。

Java SDK不会返回该参数。调用失败会抛出异常，异常信息为status_code和message的内容。

request_id string

本次调用的唯一标识符。

Java SDK返回参数为requestId。

code string

错误码，调用成功时为空值。

只有Python SDK返回该参数。

output object

调用结果信息。

属性



usage map

本次chat请求使用的Token信息。

属性



错误信息更新时间：2025-12-26 10:52:08

产品详情



我的收藏

本文介绍使用阿里云百炼服务可能出现的错误信息及解决方案。

使用阿里云 AI 助理

推荐您通过阿里云 AI 助理排查错误，输入报错信息即可得到解决方案。

示例问题：



报错信息：'code': 'Arrearage', 'param': None, 'message': 'Access denied, please make sure your account is in good standing.', 'type': 'Arrearage'}帮我看下为啥

AI 助理准确分析出原因，并给出解决方案：



400-InvalidParameter

parameter.enable_thinking must be set to false for non-streaming calls/parameter.enable_thinking only support stream call

原因： 使用非流式输出方式调用了思考模式模型。

解决方案：请将enable_thinking参数设置为false，或者改用流式输出方式调用思考模式模型。

The thinking_budget parameter must be a positive integer and not greater than xxx

原因： thinking_budget 参数不在可选值范围内。

解决方案： 请参见模型列表中模型的最大思维链长度，指定为大于0且不超过该长度的值。

This model only support stream mode, please enable the stream parameter to access the model.

原因： 模型仅支持流式输出，但调用时未启用流式输出。

解决方案： 请使用流式输出方式调用模型。

This model does not support enable_search.

原因： 当前模型不支持联网搜索能力，但指定了enable_search参数为true。

解决方案： 请调用支持联网搜索能力的模型。

暂时不支持当前设置的语种！

原因： 使用 Qwen-MT 模型时，传入的 source_lang 或 target_lang 格式错误，或不在支持的语言里。

解决方案： 请传入正确格式的英文名或语种编码。

The incremental_output parameter must be "true" when enable_thinking is true

原因： 模型开启思考模式时仅支持增量流式输出，未将incremental_output参数设置为true。

解决方案： 请将incremental_output参数设置为true再调用，API将返回增量内容。

The incremental_output parameter of this model cannot be set to False.

原因： 模型仅支持增量流式输出，未将incremental_output参数设置为true。

解决方案： 请将incremental_output参数设置为true再调用，API将返回增量内容。

Range of input length should be [1, xxx]

原因： 调用模型时输入内容长度超过模型上限。

解决方案：

若通过代码调用，请控制 messages 数组中的 Token 数在模型最大输入Token范围内；

使用对话客户端（如Chatbox）或阿里云百炼控制台进行连续对话时，每次请求都会附带历史记录，容易超出模型限制。超出限制后，请开启新对话。

Range of max_tokens should be [1, xxx]

原因： max_tokens 参数设置未在 [1, 模型最大输出 Token 数]的范围内。

解决方案： 请参考模型最大输出 Token 设置max_tokens参数。

Temperature should be in [0.0, 2.0)/'temperature' must be Float

原因： temperature参数设置不在[0.0, 2.0)范围。

解决方案： 将temperature参数设置为大于等于0，小于2的数字。

Range of top_p should be (0.0, 1.0]/'top_p' must be Float

原因： top_p参数设置不在(0.0, 1.0]范围。

解决方案： 将top_p参数设置为大于0，小于等于1的数字。

Parameter top_k be greater than or equal to 0

原因： top_k参数设置为小于0的数字。

解决方案： 将top_k参数设置为大于等于0的数字。

Repetition_penalty should be greater than 0.0

原因： repetition_penalty参数设置为小于等于0的数字。

解决方案： 将repetition_penalty参数设置为大于0的数字。

Presence_penalty should be in [-2.0, 2.0]

原因： presence_penalty参数不在[-2.0,2.0]区间。

解决方案： 将presence_penalty参数设置在[-2.0,2.0]区间。

Range of n should be [1, 4]

原因： n 参数设置未在 [1, 4]的范围内。

解决方案： 将 n 参数设置在[1, 4]范围内。

Range of seed should be [0, 9223372036854775807]

原因： 使用DashScope协议时，seed 参数设置未在 [0, 9223372036854775807]的范围内。

解决方案： 将seed参数设置在 [0, 9223372036854775807]的范围内。

Request method 'GET' is not supported.

原因： 当前接口不支持 GET 请求方法。

解决方案： 请查阅接口文档，使用该接口支持的请求方法（如 POST 等）重新发起请求。

messages with role "tool" must be a response to a preceeding message with "tool_calls"

原因： 在工具调用时没有向 messages 数组添加 Assistant Message。

解决方案： 请将模型第一轮响应的 Assistant Message 添加到 messages 数组后再添加 Tool Message。

Required body invalid, please check the request body format.

原因： 请求体（body）格式不符合接口要求。

解决方案： 请检查请求体，确保为标准的JSON字符串。常见问题有：多了,、括号未闭合等。可借助阿里云AI助理帮助修复请求体格式。

input content must be a string.

原因： 纯文本模型不支持将 messages 中的 content 设置为非字符串类型。

解决方案： 请勿将content设置为如[{"type": "text","text": "你是谁？"}]的数组类型。

The content field is a required field.

原因： 发起请求时，未指定content参数，如{"role": "user"}。

解决方案： 请指定content参数。如{"role": "user","content": "你是谁"}。

current user api does not support http call.

原因：当前模型不支持非流式输出。

解决方案：请使用流式输出。

Either \"prompt\" or \"messages\" must exist and cannot both be none

原因： 调用大模型时，既未指定messages参数，也未指定prompt参数（即将废弃）。如果指定了messages参数后报错，可能是因为格式错误，例如通过DashScope-HTTP时，messages需放入input对象中，而不是与model参数并列。

解决方案： 请指定messages参数。如果已指定但仍报错，请参见通义千问API文档，检查其位置是否正确。

'messages' must contain the word 'json' in some form, to use 'response_format' of type 'json_object'.

原因： 使用结构化输出时，提示词中不包含 json 关键词。

解决方案： 在提示词中加入json（不区分大小写），如：“请以json格式输出”。

Json mode response is not supported when enable_thinking is true

原因： 使用结构化输出时开启了模型的思考模式。

解决方案： 请在使用结构化输出时，将enable_thinking设为false关闭思考模式。也可参见常见问题思考模式模型如何结构化输出？

Tool names are not allowed to be [search]

原因： 工具名称无法设置为search。

解决方案： 工具名称请设置为search之外的值。

Unknown format of response_format, response_format should be a dict, includes 'type' and an optional key 'json_schema'. The response_format type from user is xxx.

原因： 指定的response_format参数不符合规定。

解决方案： 如需使用结构化输出功能，请将response_format参数设置为{"type": "json_object"}。

The value of the enable_thinking parameter is restricted to True.

原因： 部分模型（如qwen3-235b-a22b-thinking-2507）不可将enable_thinking参数设为 false。

解决方案：

若通过第三方工具调用（如 Cherry Studio），请打开输入框的思考开关。

若通过代码调用，请将enable_thinking设为true。

'audio' output only support with stream=true

原因： 在使用Qwen-Omni模型时，未使用流式输出方式，而模型仅支持流式输出方式。

解决方案： 设置stream参数为true以启用流式输出。

tool_choice is one of the strings that should be ["none", "auto"]

原因： 发起 Function Calling 时指定的 tool_choice 参数有误。

解决方案： 请指定为 "auto"（由大模型自主选择工具）或 "none"（强制不使用工具）。

Model not exist.

原因： 设置的model参数不存在。

解决方案： 可能是model参数大小写有误，或阿里云百炼没有您需要调用的模型。请对照模型列表中的模型名称，检查输入的model是否正确。

请勿混用开源社区的模型名与百炼模型ID，如应该使用qwen3-235b-a22b-instruct-2507，而非Qwen/Qwen3-235B-A22B-Instruct-2507。



The result_format parameter must be \"message\" when enable_thinking is true

原因： 调用思考模式模型，result_format参数未设置为"message"。

解决方案： 将result_format参数设置为"message"。

The audio is empty

原因： 输入音频时间过短，导致采样点不足。

解决方案： 请增加音频的时间。

File parsing in progress, please try again later.

原因：使用 Qwen-Long 模型时，文件未完成解析。

解决方案：请等待文件解析完成后再重试。

The "stop" parameter must be of type "str", "list[str]", "list[int]", or "list[list[int]]", and all elements within the list must be of the same type.

原因： stop 参数不符合str, list[str], list[int], 或list[list[int]]格式。

解决方案： 参见通义千问API 文档，设置正确格式的stop 参数。

Value error, batch size is invalid, it should not be larger than xxx.

原因： 调用 Embedding 模型时，文本数量超过模型上限。

解决方案： 参考Embedding文档中模型的批次大小信息，控制传入文本的数量。

Invalid file [id:file-fe-**********].

原因：提供的 file-id 无效。例如输入错误、使用了不属于当前阿里云账号的 file-id。

解决方案：通过OpenAI兼容-File确认file-id是否有效，或重新OpenAI兼容-File来获取新的file_id后进行调用。

[] is too short

原因： 输入的messages为空数组。

解决方案： 请添加 message 后再发起请求。

The tool call is not supported.

原因： 使用的模型不支持传入tools参数。

解决方案： 请更换为支持Function Calling的Qwen或DeepSeek模型。

Required parameter(xxx) missing or invalid, please check the request parameters.

原因： 接口调用参数不合法。

解决方案： 请检查请求参数，确保所有必需参数都已提供且格式正确。

input must contain file_urls

原因： 使用语音识别（Paraformer）的录音文件识别时，未对请求参数file_urls赋值。

解决方案： 请在请求中包含file_urls参数并为其赋值。

The provided URL does not appear to be valid. Ensure it is correctly formatted.

原因： 当使用视觉理解、全模态或音频理解模型时，传入数据的 URL 或本地路径无效或不符合要求。

解决方案：

传入 URL ：需要以 http://、 https://、data:开头。若以data:开头, 在 Base64 编码数据前需要包含"base64"。

传入本地路径：需要以file://开头。

传入临时URL：

通过 HTTP 调用，需确保请求的 Header 中添加了参数 X-DashScope-OssResourceResolve: enable。

通过 SDK 调用：仅支持 DashScope SDK调用，请勿使用 OpenAI SDK。

Input should be a valid dictionary or instance of GPT3Message

原因： messages 字段的构造格式不符合要求，例如括号数量不匹配、缺少必要的键值对等。

解决方案： 请检查messages字段的JSON结构是否正确。

Value error, contents is neither str nor list of str.: input.contents

原因： 使用 Embedding 模型时，输入不是字符串也不是字符串数组。

解决方案： 请修改输入格式为字符串或字符串列表。

File [id:file-fe-xxx] format is not supported.

原因： Qwen-Long模型仅限于处理纯文本格式文件（TXT、DOCX、PDF、EPUB、MOBI、MD），不支持图片或扫描文档。

解决方案： 如需对图片内容进行文本提取、分析和总结，可使用通义千问VL模型。

File [id:file-fe-**********] cannot be found.

原因： 仅在Qwen-Long模型的对话场景中，在发起对话请求后的极短时间内调用OpenAI文件兼容接口删除相关文件时才会出现。

解决方案： 请等待模型完成对话后再删除相关文件。

Too many files provided.

原因： 提供的file-id数量超限。

解决方案： 请确保file-id数量小于100。

File [id:file-fe-**********] exceeds size limit.

原因：文件大小超出限制。

解决方案： 确保文件小于150 MB。

File [id:file-fe-**********] exceeds page limits (15000 pages).

原因： 文件页数超出限制。

解决方案： 确保文件页数少于15000页。

File [id:file-fe-**********] content blank.

原因： 文件内容为空。

解决方案： 确保文件内容不为空。

Total message token length exceed model limit (10000000 tokens).

原因： 输入总长度超过了10,000,000 Token。

解决方案： 请确保message长度符合要求。

The video modality input does not meet the requirements because: the range of sequence images shoule be (4, 512)./(4,80).

原因： 使用通义千问 VL 模型以图像列表方式输入视频时，图像数量不符合要求。

解决方案： Qwen3-VL与Qwen2.5-VL系列模型需传入4-512张图片；其他模型需传入4-80张图片。详情可参见视觉理解。

Exceeded limit on max bytes per data-uri item : 10485760'. / Multimodal file size is too large

原因： 向多模态模型（Qwen-VL、QVQ、Qwen-Omni）传入的本地图像或视频超出大小限制。

解决方案：

本地文件：Base64 编码后单个文件不得超过 10 MB。

文件 URL：图像文件不得超过 10 MB；对于视频文件，

Qwen3-VL、qwen-vl-max、qwen-vl-max-latest、qwen-vl-max-2025-08-13、qwen-vl-max-2025-04-08：不超过 2GB；

qwen-vl-plus系列及qwen-vl-max-2025-04-08之前的更新的模型：不超过 1GB；

其他模型不超过 150MB。

压缩文件体积请参见如何将图像或视频压缩到满足要求的大小？

Input should be 'Cherry', 'Serena', 'Ethan' or 'Chelsie': parameters.audio.voice

原因： 使用Qwen-Omni或Qwen-TTS 时voice参数指定错误。

解决方案： 请指定为'Cherry', 'Serena', 'Ethan' 或 'Chelsie'中的一个。

The image length and width do not meet the model restrictions.

原因： 传入通义千问VL模型的图像尺寸（长度和宽度）不符合模型的要求。

解决方案： 图像尺寸需满足以下要求：宽度和高度均不小于10像素，且宽高比不应超过200:1或1:200。

Failed to decode the image during the data inspection.

原因： 图像解码失败。

解决方案： 请确认图像是否有损坏，以及图像格式是否符合要求。

The file format is illegal and cannot be opened. / The audio format is illegal and cannot be opened. / The media format is not supported or incorrect for the data inspection.

原因： 无法支持的文件格式或文件无法打开。

解决方案： 请确认文件是否损坏、文件扩展名和实际格式是否匹配、文件格式是否支持。

The input messages do not contain elements with the role of user.

原因：

调用模型时，未向模型传入 User Message；

或API调用阿里云百炼工作流应用时，在开始节点中传入的参数，需通过biz_params参数传递（而非user_prompt_params）。

解决方案： 请确保向模型传入User Message，或正确传递自定义参数。

Failed to download multimodal content. / Download the media resource timed out during the data inspection process. / Unable to download the media resource during the data inspection process.

原因：服务端无法下载公网 URL 指向的媒体文件，可能由以下原因导致。

连通性问题： 使用了阿里云对象存储服务内网地址。

网络延迟： 跨地域访问引发超时。

服务不稳定： 源存储服务响应慢或不可达。

解决方案：

更换存储服务

建议使用与模型服务同地域的存储服务。推荐使用阿里云对象存储服务生成公网链接（请勿使用内网地址）。

调整传输方式

若传入公网 URL 方式失败，可参考传入本地文件（Base64 编码或文件路径）切换为推荐的传入方式：









文件类型

文件规格

DashScope SDK（Python、Java）

OpenAI 兼容 / DashScope HTTP

图像

大于 7MB 小于 10MB

传入本地路径

仅支持公网 URL，建议使用阿里云对象存储服务

小于 7MB

传入本地路径

Base64 编码

视频

大于 100 MB

仅支持公网 URL，建议使用阿里云对象存储服务

仅支持公网 URL，建议使用阿里云对象存储服务

大于 7MB 小于 100 MB

传入本地路径

仅支持公网 URL，建议使用阿里云对象存储服务

小于 7MB

传入本地路径

Base64 编码

音频

大于 10 MB

仅支持公网 URL，建议使用阿里云对象存储服务

仅支持公网 URL，建议使用阿里云对象存储服务

大于 7MB 小于 10 MB

传入本地路径

仅支持公网 URL，建议使用阿里云对象存储服务

小于 7MB

传入本地路径

Base64 编码

Base64 编码会增大数据体积，原始文件大小应小于 7 MB。

使用 Base64 或本地路径可避免服务端下载超时，提升稳定性。

url error, please check url！

原因： 有以下两种可能情况：

DashScope SDK 版本过低：调用图像/视频生成模型时，旧版SDK无法识别正确的服务端地址；

接口与模型不匹配：例如用多模态接口（如MultiModalConversation.call()）调用 qwen3-max 等纯文本模型；或在直接发送 POST 请求时填写了错误的 URL。

解决方案：

升级SDK版本：若调用图像/视频生成模型，请升级 SDK 版本。

查阅 API 文档：容易触发此错误的模型与对应的接口如下：

多模态模型（如qwen3-vl-plus、qwen-vl-max）：MultiModalConversation.call()与POST https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation；

若使用 spring-ai-alibaba 框架，请确认是否设置多模态参数withMultiModel。

相关文档：视觉理解。

纯文本模型（如qwen3-max、qwen-plus、deepseek-v3.2）：Generation.call()与POST https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation。

相关文档：文本生成模型概述。

CosyVoice声音复刻接口：该接口包含 model 与 target_model 参数：需将 model 设置为 voice-enrollment， target_model 设置为具体的 CosyVoice 模型。

相关文档：CosyVoice声音复刻API

请查阅对应模型的 API 文档，使用匹配的接口。

Don't have authorization to access the media resource during the data inspection process.

原因： 调用模型时，传入的OSS中带签名的文件URL已经过期。

解决方案： 请确保在文件URL的有效期内访问该文件。

The item of content should be a message of a certain modal.

原因： 使用DashScope SDK调用多模态模型时，content 数组中每个元素的键必须为以下值之一：image、video、audio 或 text。

解决方案： 请并使用正确的content参数。

Invalid video file.

原因： 传入的视频文件无效。

解决方案： 请检查视频文件是否损坏或格式是否正确。

The video modality input does not meet the requirements because: The video file is too long.

原因： 传入通义千问VL模型或者Qwen-Omni 模型的视频时长超过限制。

解决方案：

Qwen2.5-VL模型支持的视频时长应在2秒至10分钟之间。

其他通义千问VL或Qwen-Omni 模型支持的视频时长应在2秒至40秒之间。

Field required: xxx

原因： 缺少入参。

解决方案： 请根据错误提示xxx补充对应的参数。

The request is missing required parameters or in a wrong format, please check the parameters that you send.

原因： 缺少入参，或入参格式错误。

解决方案： 请检查请求参数是否完整且格式正确。

Invalid ext_bbox.

原因： 输入的ext_bbox无效。

解决方案： 详情参见Emoji 视频生成。

Driven not exist: driven_id.

原因： 输入的driven_id不存在。

解决方案： 详情参见Emoji 视频生成。

Missing training files.

原因： 参数错误，缺少参数或者参数格式问题等。

The style is invalid.

原因： style不在枚举范围内。

解决方案： 请检查style参数的取值是否正确。

The style_level is invalid.

原因： style_level不在枚举范围内。

解决方案： 详情参见EMO 视频生成。

parameters.video_ratio must be 9:16 or 3:4.

原因： video_ratio 入参只能为 9:16 或 3:4。

解决方案： 请修改video_ratio参数为 "9:16" 或 "3:4"。

the xxx parm is invalid!

原因： 输入参数超出范围。

解决方案： 详情参见视频风格重绘。

input json error.

原因： 输入JSON错误。

解决方案： 请检查请求的JSON格式是否正确。

read image error.

原因： 读取图像失败。

解决方案： 请检查图像文件是否损坏或格式是否正确。

the parameters must conform to the specification: xxx.

原因： 输入参数值超出范围。

解决方案： 请根据错误提示xxx检查并修正参数值。

The size of person image and coarse_image are not the same.

原因： coarse_image分辨率和person_image不一致。

解决方案： 请确保coarse_image和person_image的分辨率一致。

The request is missing required parameters or the parameters are out of the specified range, please check the parameters that you send.

原因： 缺少必要的接口调用参数或参数越界。

解决方案： 请检查并修正请求参数。

image format error

原因： 图片格式错误。

解决方案： 需要是图片url或者Base64字符串。

No messages found in input

原因： 请求参数中需要有messages字段。

解决方案： 详情参见通义千问-图像编辑。

Invalid image format or corrupted file

原因： 输入图片格式错误或文件损坏。

解决方案： 请检查文件是否可正常打开和下载，确保文件完整且格式符合要求。

download image failed

原因： 图片不能下载。

解决方案： 请检查文件是否可正常下载。

messages length only support 1

原因： messages数组长度仅支持 1。

解决方案： 即只能传入一条对话消息。详情参见通义千问-图像编辑。

content length only support 2

原因： content数组长度仅支持为2。

解决方案： 即只能传入一组text和image。详情参见通义千问-图像编辑。

lack of image or text

原因： 请求参数缺少image或text字段。

解决方案： 详情参见通义千问-图像编辑。

num_images_per_prompt must be 1.

原因： 请求参数不合法，参数n（生成图片数量）只能设置为1。

解决方案： 请将参数n的值设置为1。

Input files format not supported.

原因： 音频、图片格式不符合要求。

解决方案： 音频支持格式mp3, wav, aac，图片支持格式jpg, jpeg, png, bmp, webp。详情参见LivePortrait 视频生成。

Failed to download input files.

原因： 输入文件下载失败。

解决方案： 请检查文件URL是否可访问，网络是否通畅。

oss download error.

原因： 输入图像下载失败。

解决方案： 请检查图像的OSS链接是否正确且可访问。

The image content does not comply with green network verification.

原因： 图像内容不合规。

解决方案： 请更换符合内容安全规范的图像。

read video error.

原因： 读取视频失败。

解决方案： 请检查视频文件是否损坏或格式不受支持。

the size of input image is too small or too large.

原因： 输入图像的尺寸过小或者过大。

解决方案： 请调整图像尺寸以符合API要求。

The request parameter is invalid, please check the request parameter.

原因： clothes_type入参不合规。

解决方案： 详情参见AI试衣-图片分割。

The type or value of {parameter} is out of definition.

原因： 参数类型或值不符合要求。

解决方案： 详情参见LivePortrait 视频生成。

The request parameter is invalid, please check the request parameter.

原因： 画幅入参不合规。

解决方案： 可选"1:1"或"3:4"。

request timeout after 23 seconds.

原因： 超过23秒未向服务发送数据。该报错信息在使用实时语音合成（Sambert）、实时语音识别/翻译（Gummy）、语音识别（Paraformer）和实时语音合成（CosyVoice）时产生。

解决方案： 请检查为什么长时间未向服务器发送数据。如果长时间（超过23秒）不向服务端发送消息，请及时结束任务。

Please ensure input text is valid.

原因： 若您使用实时语音合成（CosyVoice），此错误通常是由于未发送待合成文本引起的。可能原因包括：参数遗漏（未为 text 参数赋值）或代码异常（导致对 text 参数的赋值失败）。

解决方案： 请排查代码，确保 text 参数被正确赋值并发送。

Missing required parameter 'payload.model'! Please follow the protocol!

原因： 若您使用实时语音合成（CosyVoice），此错误通常是由于发送run-task指令未指定model参数。

解决方案： 请指定 model 参数。

[tts:]Engine return error code: 418

原因： 使用实时语音合成（CosyVoice），请求参数 voice（音色）不正确，或 model（模型）与 voice（音色）版本不匹配。

解决方案：

检查 voice 参数赋值：

如果使用的是默认音色，请对照Python SDK中的“voice参数”进行确认。

如果使用的是声音复刻音色，请通过查询指定音色接口确认音色状态为“OK”，并确保音色归属账号与调用账号一致。

检查版本匹配：v2模型只能使用v2的音色，v1模型只能使用v1的音色，两者不可混用。

Request voice is invalid!

原因： 若您使用实时语音合成（CosyVoice），此错误通常是因为未设置音色。

解决方案： 请检查是否对voice参数赋值。若您使用WebSocket API，请参照API文档按照正确JSON格式配置参数。

ref_images_url and obj_or_bg must be the same length.

原因： 使用通义万相-通用视频编辑的多图参考功能时，ref_images_url和obj_or_bg的数组长度不一致。

解决方案： 请确保ref_images_url和obj_or_bg的数组长度一致。

check input data style.

原因： 输入参数不满足入参要求。

解决方案： 请检查并修正输入参数。

An error during model pre-process.

原因： 传入了错误格式的 content 字段。

解决方案：

若通过代码调用，请勿将content设置为如[{"type": "text", "text": "你是谁？"}]的array类型。

若使用 Cline ，请在设置界面单击MODEL CONFIGURATION，并勾选 Enable R1 messages format。

The image size is not supported for the data inspection.

原因：

传入通义千问VL模型的图像尺寸（长度和宽度）不符合模型的要求。

输出图像大小超出限制（10MB）。

解决方案：

图像尺寸需满足以下要求：

图像的宽度和高度均不小于10像素。

宽高比不应超过200:1或1:200

调整生成图像的参数。

Required parameter(data_sources) missing or invalid, please check the request parameters.

原因： 调用 SubmitIndexJob 接口时返回此错误，原因是调用 CreateIndex 接口时未指定必传参数SourceType。

解决方案： 基于给定文档创建知识库时，此参数需传入DATA_CENTER_FILE；基于给定类目创建知识库时，此参数需传入DATA_CENTER_CATEGORY。详见CreateIndex文档。

Wrong Content-Type of multimodal url

原因：URL请求的响应头信息Content-Type字段不正确。

通义千问VL模型支持的Content Type为：image/bmp、image/bmp、image/icns、image/x-icon、image/jpeg、image/jp2、image/png、image/sgi、image/tiff、image/webp。详情可参见通义千问VL模型支持的图像。

解决方案：

查看Content-Type字段

Field required: image_url

原因： 缺少入参image_url。

解决方案： 请参考Emoji 视频生成，传入image_url参数。

Field required: driven_id

原因： 缺少入参driven_id。

解决方案： 请参考Emoji 视频生成，传入driven_id参数。

Invalid ext_bbox

原因： 输入ext_bbox参数无效。

解决方案： 请参考Emoji 视频生成，传入正确的ext_bbox。

Driven not exist: driven_id

原因：输入driven_id不存在。

解决方案： 请参考Emoji 视频生成，传入正确的driven_id。

Text request limit violated, expected 1.

原因：在调用CosyVoice语音合成的WebSocket API时，将enable_ssml设为true后多次发送continue-task指令。

解决方案： enable_ssml设为true后，只允许发送一次continue-task指令。

400-invalid_request_error-invalid_value

-1 is lesser than the minimum of 0 - 'seed'/'seed' must be Integer

原因： 使用OpenAI兼容协议时，seed 参数设置未在 [0, 231-1]的范围内。

解决方案： 将seed参数设置在 [0, 231-1]的范围内。

400-invalid_request_error

you must provide a model parameter.

原因： 请求时没有提供 model 参数。

解决方案： 请在请求中添加model参数。

400-InvalidParameter.NotSupportEnableThinking

The model xxx does not support enable_thinking.

原因： 当前使用的模型不支持设定参数 enable_thinking。

解决方案： 请求时去掉enable_thinking参数，或使用支持思考模式的模型。

400-invalid_value

The requested voice 'xxx' is not supported.

原因： 在进行Qwen-TTS实时语音合成时，选用的音色是通过Qwen-TTS声音复刻功能生成的，但二者使用的模型不同。

解决方案： 请检查声音复刻时的请求参数target_model和语音合成时的请求参数model是否一致。

400-Arrearage

Access denied, please make sure your account is in good standing.

原因： API Key 所属的阿里云账号存在欠费，导致访问被拒绝。

解决方案：前往费用与成本查看是否欠费：

未欠费：请确认该 API Key 是否属于当前账号；

欠费：请及时充值。充值后，系统余额可能存在延迟，请稍等后重试。

说明

Q：购买了资源包，且只调用了资源包内的模型，为什么会欠费？

A：请核对资源包的可抵扣范围。以qwen-plus/qwen-plus-latest系列资源包为例：

资源包仅可抵扣qwen-plus、qwen-plus-latest模型input tokens和output tokens费用，且仅支持抵扣单次请求输入在0<Token≤128K阶梯范围内产生的实时推理费用（非思考模式）

不支持抵扣的费用包括：

单次请求输入在Token>128K阶梯范围产生的费用。

Batch调用、上下文缓存、模型调优、模型部署产生的费用。

400-DataInspectionFailed/data_inspection_failed

Input or output data may contain inappropriate content. / Input data may contain inappropriate content. / Output data may contain inappropriate content.

原因： 输入或者输出包含疑似敏感内容被绿网拦截。

解决方案： 请修改输入内容后重试。

Input xxx data may contain inappropriate content.

原因： 输入数据（如提示词或图像）可能包含敏感内容。 解决方案： 内容合规检查，请修改输入后重试。

400-APIConnectionError

Connection error.

原因： 本地网络问题，通常是因为开启了代理。

解决方案： 请关闭或者重启代理。

400-InvalidFile.DownloadFailed

The audio file cannot be downloaded.

原因： 使用语音识别（Paraformer）录音文件识别，待识别文件下载失败。

解决方案： 请检查待识别音频文件URL是否可通过公网访问。

400-InvalidFile.AudioLengthError

Audio length must be between 1s and 300s.

原因： 音频长度不符合要求。

解决方案： 请确保音频时长在[1, 300]秒范围内

Audio length must be between 1s and 180s.

原因： 音频长度不符合要求。

解决方案： 请确保音频时长在[1, 180]秒范围内。

400-InvalidFile.NoHuman

The input image has no human body. Please upload other image with single person.

原因： 输入图片中没有人或未检测到人脸。

解决方案： 请上传单人照。

400-InvalidFile.BodyProportion

The proportion of the detected person in the picture is too large or too small, please upload other image.

原因： 上传图片中人物占比不符合要求。

解决方案： 请上传符合人物占比要求的图片。

400-InvalidFile.FacePose

The pose of the detected face is invalid, please upload other image with whole face and expected orientation.

原因： 上传图片中人物面部姿态不符合要求（要求面部可见，头部朝向无严重偏移）。

解决方案： 请上传符合要求的图片。

The pose of the detected face is invalid, please upload other image with the expected oriention.

原因： 上传图片中人物面部姿态不符合要求（要求面部朝向无严重偏移）。

解决方案： 请确保图片中人脸朝向无偏斜。

The pose of the detected face is invalid, please upload other image with the expected orientation.

原因： 上传图片中人物面部姿态不符合要求（要求面部朝向无严重偏移）。

解决方案： 请确保图片中人脸朝向无偏斜。

400-InvalidFile.Resolution

The image resolution is invalid, please make sure that the largest length of image is smaller than 7000, and the smallest length of image is larger than 400.

原因： 上传图像大小不符合要求。

解决方案： 上传图片的分辨率不得高于7000*7000，且不得低于400*400。

The image resolution is invalid, please make sure that the largest length of image is smaller than 4096, and the smallest length of image is larger than 224.

原因： 上传图像大小不符合要求。

解决方案： 上传图片的分辨率最长边小于 4096 像素，且最短边大于 224 像素。

The image resolution is invalid, please make sure that the largest length of image is smaller than xxx, and the smallest length of image is larger than yyy.

原因： 上传图像大小不符合要求。

解决方案： 上传图片的分辨率不得高于xxx*xxx，且不得低于yyy*yyy。

The image resolution is invalid, please make sure that the aspect ratio is smaller than xxx, and largest length of image is smaller than yyy.

原因： 上传图像大小不符合要求。

解决方案： 上传图片的长宽比必须小于xxx，且分辨率不得高于yyy*yyy。

Invalid video resolution. The height or width of video must be xxx ~ yyy.

原因： 视频分辨率不符合要求。

解决方案： 视频边长需介于xxx-yyy之间。

400-InvalidFile.FPS

Invalid video FPS. The video FPS must be 15 ~ 60.

原因： 视频帧率不符合要求。

解决方案： 视频帧率需介于15-60fps之间。

400-InvalidFile.Value

The value of the image is invalid, please upload other clearer image.

原因： 上传图片过暗不符合要求。

解决方案： 请确保图片中人脸清晰。

400-InvalidFile.FrontBody

The pose of the detected person is invalid, please upload other image with the front view.

原因： 上传图片中人物背身不符合要求。

解决方案： 请确保图片中人物正面朝向镜头。

400-InvalidFile.FullFace

The pose of the detected face is invalid, please upload other image with whole face.

原因： 上传图片中人物面部姿态不符合要求（要求面部可见）。

解决方案： 请确保图片中人脸完整无遮挡。

400-InvalidFile.FaceNotMatch

There are no matched face in the video with the provided reference image.

原因： 参考图与视频人脸匹配失败。

解决方案： 详情参见VideoRetalk视频生成。

400-InvalidFile.Content

The first frame of input video has no human body. Please choose another clip.

原因： 视频首帧需要有人。

解决方案： 请选择包含人体的视频片段。

The human is too small in the first frame of input video. Please choose another clip.

原因： 视频首帧人物过小。

解决方案： 请选择首帧人物占比较大的视频。

The human is not clear in the first frame of input video. Please choose another clip.

原因： 视频首帧人物不清晰。

解决方案： 请选择首帧人物清晰的视频。

The input image has no human body or multi human bodies. Please upload other image with single person.

原因： 输入图片中没有人或有多人。

解决方案： 请上传单人照。

The input image has no human body or has unclear human body. Please upload other image.

原因： 输入图片中人体不完整或者没有人体。

解决方案： 请上传包含完整清晰人体的图片。

The input image has multi human bodies. Please upload other image with single person.

原因： 输入图片中有多人。

解决方案： 请上传单人照。

400-InvalidFile.FullBody

The human is not fullbody in the first frame of input video. Please choose another clip.

原因： 视频首帧人物不完整。

解决方案： 需露出人物全身。

The pose of the detected person is invalid, please upload other image with whole body, or change the ratio parameter to 1:1。

原因： 上传图片中人物姿态不符合要求。

解决方案： 请上传符合要求的图片，头像照要求头部完整可见，半身照要求髋部以上完整可见，或者调整图像宽高比为1:1。

400-InvalidFile.BodyPose

The pose of the detected person is invalid, please upload other image with whole body and expected orientation.

原因： 单人的动作不符合要求。

解决方案： 请上传符合要求的图片，要求肩膀及踝部可见，非背身，非坐姿，人物朝向无严重偏移。

400-InvalidFile.Size

Invalid file size. The video file size must be less than 200MB, and the audio file size must be less than 15MB.

原因： 文件大小不符合要求。

解决方案： 视频文件必须小于200MB，音频文件必须小于15MB。

Invalid file size, The image file size must be smaller than 5MB.

原因： 文件大小不符合要求。

解决方案： 图片文件必须小于5MB。

Invalid file size. The video/audio/image file size must be less than xxxMB.

原因： 文件大小不符合要求。

解决方案： 视频/音频/图像文件必须小于指定的MB数。

400-InvalidFile.Duration

Invalid file duration. The file duration must be xxx s ~ yyy s.

原因： 文件时长不符合要求。

解决方案： 视频/音频文件时长需要介于xxx-yyy s之间。

400-InvalidFile.ImageSize

The size of image is beyond limit.

原因： 图片大小超出限制。

解决方案： 要求图片长宽比例不大于2，且最长边不大于4096。

400-InvalidFile.AspectRatio

Invalid file ratio. The file aspect ratio (height/width) must be between 3:1 and 1:3.

原因： 文件长宽比不符合要求。

解决方案： 视频文件长宽比需要介于3:1到1:3之间。

Invalid file ratio. The file aspect ratio (height/width) must be between 2.0 and 0.5.

原因： 文件长宽比不符合要求。

解决方案： 图片文件宽高比必须在2.0到0.5之间。

400-InvalidFile.Openerror

Invalid file, cannot open file as video/audio/image.

原因： 文件无法打开。

解决方案： 请检查文件是否损坏或格式是否正确。

400-InvalidFile.Template.Content

Invalid template content.

原因： 动作模板无权限，或模板内容不符合要求。

解决方案： 请检查模板权限和内容。

400-InvalidFile.Format

Invalid file format，the request file format is one of the following types: MP4, AVI, MOV, MP3, WAV, AAC, JPEG, JPG, PNG, BMP, and WEBP.

原因： 文件格式不符合要求。

解决方案： 使用符合要求的文件：视频支持mp4、avi、mov；音频支持mp3, wav, aac；图片支持jpg, jpeg, png, bmp, webp。

400-InvalidFile.MultiHuman

The input image has multi human bodies. Please upload other image with single person.

原因： 输入图片中有多人。

解决方案： 请上传单人照。

400-InvalidPerson

The input image has no human body or multi human bodies. Please upload other image with single person.

原因： 输入图片中没有人或有多人。

解决方案： 请上传单人照。

400-InvalidParameter.DataInspection

Unable to download the media resource during the data inspection process.

原因： 下载图片或音频文件超时。

解决方案： 如果从海外发起调用，由于跨境网络不稳定，可能会导致下载资源超时。请将文件存储到国内的 OSS 中，再发起模型调用。也可以使用临时存储空间上传文件。

400-FlowNotPublished

Flow has not published yet, please publish flow and try again.

原因： 流程未发布。

解决方案： 请发布流程后再重试。

400-InvalidImage.ImageSize

The size of image is beyond limit.

原因： 图片大小超出限制。

解决方案： 要求图片长宽比例不大于2，且最长边不大于4096。

400-InvalidImage.NoHumanFace

No human face detected.

原因： 未检测到人脸（仅生成任务异步查询接口）。

解决方案： 请上传包含清晰人脸的图片。

400-InvalidImageResolution

The input image resolution is too large or small.

原因： 输入图像分辨率过大或过小。

解决方案： 图像分辨率不低于256×256像素，不超过5760×3240像素。

400-InvalidImageFormat

The input image is in invalid format.

原因： 图片格式不符合要求。

解决方案： 使用JPEG、PNG、JPG、BMP、WEBP格式的图片。

400-InvalidURL

Invalid URL provided in your request.

原因： URL 无效。

解决方案： 使用有效的 URL。

Required URL is missing or invalid, please check the request URL.

原因： 输入的URL无效或缺失。

解决方案： 请提供正确的URL。

The request URL is invalid, make sure the url is correct and is an image.

原因： 输入的URL无效。

解决方案： 请确保URL正确且指向一个图像文件。

The input audio is longer than xxs.

原因： 输入的音频文件超过最大时长xx秒。

解决方案： 请将音频文件裁剪至xx秒以内。

File size is larger than 15MB.

原因： 输入的音频文件超过最大限制15MB。

解决方案： 请将音频文件压缩至15MB以内。

File type is not supported. Allowed types are: .wav, .mp3.

原因： 输入的音频格式不合规。

解决方案： 当前仅支持wav、mp3格式。

The request URL is invalid, please check the request URL is available and the request image format is one of the following types: JPEG, JPG, PNG, BMP, and WEBP.

原因： 图片不可访问或下载的文件格式不支持。

解决方案： 请确保URL可访问，且图片格式为JPEG, JPG, PNG, BMP或WEBP。

400-InvalidImage.FileFormat

Invalid image type. Please ensure the uploaded file is a valid image.

原因：图片文件格式不支持。

解决方案：使用JPG、JPEG、PNG、BMP、WEBP格式的图片。

400-InvalidURL.ConnectionRefused

Connection to xxx refused, please provide available URL.

原因： 下载被拒绝。

解决方案： 请提供可用的URL。

400-InvalidURL.Timeout

Download xxx timeout, please check network connection.

原因： 下载超时。

解决方案： 请检查网络连接。

400-BadRequestException

Invalid part type.

原因： 仅在Qwen-Long模型的对话场景中，用户上传了Qwen-Long模型暂不支持的文件类型。

解决方案： 请上传Qwen-Long支持的文件类型。

400-BadRequest.EmptyInput

Required input parameter missing from request.

原因： 请求时未添加input参数。

解决方案： 请在请求中添加input参数。

400-BadRequest.EmptyParameters

Required parameter "parameters" missing from request.

原因：请求时未添加 parameters参数。

解决方案： 请在请求中添加parameters参数。

400-BadRequest.EmptyModel

Required parameter "model" missing from request.

原因： 请求时未提供 model参数。

解决方案： 请在请求中添加model参数。

400-BadRequest.IllegalInput

The input parameter requires json format.

原因： 入参格式不符合API要求的JSON格式。

解决方案： 请检查入参数格式，确保为标准的JSON。

400-BadRequest.InputDownloadFailed

Failed to download the input file: xxx.

原因： 下载输入文件失败，可能是由于下载超时、下载失败或者文件超过限额大小。

解决方案： 请根据详细错误信息xxx排查。

Failed to download the input file.

原因： 使用Qwen-TTS声音复刻时，服务器下载待复刻音频失败。

解决方案： 请检查音频文件是否可以正常下载，若能下载，请检查音频文件大小是否超出限制（超过10MB）。

400-BadRequest.UnsupportedFileFormat

File format unsupported.

原因：CosyVoice声音复刻时，上传的音频格式不符合模型要求。

解决方案： 音频格式需为 WAV（16bit）、MP3 或 M4A。需要注意的是，不能仅凭文件后缀名判断格式，例如，后缀名为 .mp3 的文件可能是其他格式（如 Opus）。建议通过工具（如ffprobe、mediainfo）或命令（如Linux/macOS的file命令）确认音频文件的实际编码格式，以确保符合要求。

Input file format is not supported.

原因： 输入文件的格式不受支持。

解决方案： 请使用支持的文件格式。

400-BadRequest.TooLarge

Payload Too Large.

原因： 文件大小超出限制。

解决方案：

“purpose”参数为“file-extract”时文档不能超150MB、图片不能超20MB。

“purpose”参数为“batch”时，文件不能超500MB。 请拆分并分批上传文件。

400-BadRequest.ResourceNotExist

The Required resource not exist.

原因：

CosyVoice声音复刻更新、查询或删除接口调用时，对应音色不存在。

使用定制热词（Paraformer）或定制热词（Gummy）时，更新、查询或删除接口调用的热词资源不存在。

400-Throttling.AllocationQuota

您当前的配额为xxx

原因： CosyVoice声音复刻音色数量已达限额。

解决方案： 删除部分音色。

Free allocated quota exceeded.

原因： 使用定制热词（Paraformer）或定制热词（Gummy）时，热词数目已超过上限（每个账号默认10个，Paraformer和Gummy共用）。

解决方案： 删除部分热词。

Maximum voice storage limit exceeded, please delete existing voices.

原因： 使用Qwen-TTS声音复刻时，超过主账号可用的音色数目上限。

解决方案： 请删除一部分音色或申请扩容。

400-InvalidGarment

Missing clothing image.Please input at least one top garment or bottom garment image.

原因： 缺少服饰图片。

解决方案： 请至少提供一张上装 (top_garment_url) 或下装 (bottom_garment_url) 的图片。

400-InvalidSchema

Database schema is invalid for text2sql.

原因： 未输入数据库Schema信息。

解决方案： 请输入数据库Schema信息。

400-InvalidSchemaFormat

Database schema format is invalid for text2sql.

原因： 输入数据表信息格式异常。

解决方案： 请检查并修正数据表信息的格式。

400-Audio.AudioShortError

valid audio too short!

原因： 用于CosyVoice声音复刻的音频有效时长过短。

解决方案：音频时长应尽量控制在 10~15 秒之间。录音时请确保朗读连贯，并包含至少一段超过 5 秒的连续语音。

400-Audio.AudioSilentError

silent audio error.

原因： CosyVoice声音复刻音频文件为静音或非静音长度过短。

解决方案： 用于声音复刻的音频时长应尽量控制在 10~15 秒之间，并包含至少一段超过 5 秒的连续语音。

400-InvalidInputLength

The image resolution is invalid, please make sure that the largest length of image is smaller than 4096, and the smallest length of image is larger than 150. and the size of image ranges from 5KB to 5MB.

原因： 图片尺寸或文件大小不符合要求。

解决方案： 请参见输入图片要求。

400-FaqRuleBlocked

Input or output data is blocked by faq rule.

原因： 命中FAQ规则干预模块。

400-ClientDisconnect

Client disconnected before task finished!

原因： 任务结束前，客户端主动断开了连接。该报错信息在使用语音合成或识别相关服务时产生。

解决方案： 请检查代码，不要在任务结束前断开和服务端的连接。

400-ServiceUnavailableError

Role must be user or assistant and Content length must be greater than 0.

原因： 输入内容长度为0或role不正确。

解决方案： 请检查输入内容长度大于0，并确保参数格式（如role）符合API文档的要求。

400-IPInfringementSuspect

Input data is suspected of being involved in IP infringement.

原因： 输入数据（如提示词或图像）涉嫌知识产权侵权。

解决方案： 内容合规检查，请检查输入，确保不包含引发侵权风险的内容。

400-UnsupportedOperation

The operation is unsupported on the referee object.

原因： 关联的对象不支持该操作。

解决方案： 请检查操作对象和操作类型是否匹配。

The fine-tune job can not be deleted because it is succeeded,failed or canceled.

原因： 无法删除该微调任务，因为其状态已是“成功”、“失败”或“已取消”。

解决方案： 只有处于特定状态的任务才能被删除，请勿删除已终结状态的任务。

400-CustomRoleBlocked

Input or output data may contain inappropriate content with custom rule.

原因： 请求或响应内容没有通过自定义策略。

解决方案： 请检查内容或调整自定义策略。

400-Audio.PreprocessError

Audio preprocess error.

原因： 使用Qwen-TTS声音复刻时，待复刻音频预处理异常，可能的原因为：text参数内容与音频文本差别过大、有效人声过短、无声音等。

解决方案： 请调整text参数的内容，若调整后无效，请参照录音操作指南重新录制音频。

No segments meet minimum duration requirement

原因： 使用Qwen-TTS声音复刻时，待复刻音频有效人声过短。

解决方案： 请参照录音操作指南重新录制音频。

400-BadRequest.VoiceNotFound

Voice '%s' not found.

原因： 使用Qwen-TTS声音复刻时，调用删除音色接口时，音色已删除或音色不存在。

解决方案： 请检查传入的voice参数是否正确。

400-Audio.DecoderError

Decoder audio file failed.

原因： 使用Qwen-TTS声音复刻时，待复刻音频解码失败。/ CosyVoice声音复刻音频文件解码失败。

解决方案： 请检查音频文件是否损坏，并确保音频满足音频文件格式要求（如Qwen-TTS）或为 WAV（16bit）、MP3 或 M4A（如CosyVoice）。

400-Audio.AudioRateError

File sample rate unsupported.

原因： 使用Qwen-TTS声音复刻或CosyVoice声音复刻时，待复刻音频采样率不符合要求。

解决方案： 采样率需大于等于24000 Hz。

400-Audio.DurationLimitError

Audio duration exceeds maximum allowed limit.

原因： 使用Qwen-TTS声音复刻时，待复刻音频过长。

解决方案： 音频不得超过60秒。

401-InvalidApiKey/invalid_api_key

Invalid API-key provided. / Incorrect API key provided.

原因： API Key 填写错误。

解决方案： 常见错误原因及修正方式如下：

读取错误的环境变量

错误写法：api_key=os.getenv("sk-xxx") ，系统将尝试读取名为 sk-xxx 的环境变量，而非将 sk-xxx 当作密钥。

正确写法：

若已配置环境变量：请写为api_key=os.getenv("DASHSCOPE_API_KEY")；

确保运行前已设置DASHSCOPE_API_KEY环境变量。

若未配置环境变量：请写为api_key = "sk-xxx"。

此方式便于调试，请勿用于生产环境。

工具适配问题：第三方工具未正确适配（如Dify最新版本插件不稳定导致报错，可尝试安装旧版本通义千问插件；旧版本Cline调用模型时API Provider选择了Alibaba Qwen，应选择OpenAI Compatible）。

填写错误：阿里云百炼的 API Key 以 sk- 开头，请确认未误填其他模型提供商的密钥。

地域不匹配：API Key 和 Base URL 属于不同的地域，例如使用了中国大陆（北京）地域的 API Key 和国际（新加坡）地域的 Base URL（含 -intl）。请确认您使用的 API Key 位于北京地域页面还是新加坡地域页面，各地域对应的 Base URL 如下：







地域

OpenAI兼容

DashScope

中国大陆（北京）

https://dashscope.aliyuncs.com/compatible-mode/v1

https://dashscope.aliyuncs.com/api/v1

国际（新加坡）

https://dashscope-intl.aliyuncs.com/compatible-mode/v1

https://dashscope-intl.aliyuncs.com/api/v1

若以上均不符合，可能是 API Key 被删除，请重新获取并发起调用。

401-NOT AUTHORIZED

Access denied: Either you are not authorized to access this workspace, or the workspace does not exist. Please:\nVerify the workspace configuration.\nCheck your API endpoint settings. Ensure you are targeting the correct environment.

原因：

WorkspaceId值无效，或当前账号不是该业务空间的成员。

或者请求的接入地址（服务接入点）有误。

解决方案：

请确认WorkspaceId值无误且账号已是该业务空间的成员后，再调用接口。

中国站用户请使用华北2（北京）地域的接入地址；国际站用户请使用新加坡地域的接入地址。使用在线调试时，确认服务地址正确（如下图）。





403-AccessDenied/access_denied

Current user api does not support asynchronous calls.

原因： 接口不支持异步调用。

解决方案： 请移除请求头中的 X-DashScope-Async，或将其值设为 disable。

current user api does not support synchronous calls.

原因： 接口不支持同步调用。

解决方案： 请在请求头中设置 X-DashScope-Async: enable。

Invalid according to Policy: Policy expired.

原因： 在获取临时公网URL时，文件上传凭证已经过期。

解决方案： 请重新调用文件上传凭证接口生成新凭证。

Access denied.

原因： 无权访问此模型。可能因该模型需申请权限，或模型免费额度已耗尽且不支持付费使用（如 deepseek-r1-distill-llama-70b）。

解决方案： 请前往阿里云百炼控制台，在模型广场的对应模型卡片下方单击立即申请发起测试申请。或改用其他模型，例如通义千问或通义万相的文生图模型替代 Flux。

403-AccessDenied.Unpurchased

Access to model denied. Please make sure you are eligible for using the model.

原因： 未开通阿里云百炼服务。

解决方案： 请参照以下流程开通阿里云百炼服务。

注册账号：如果没有阿里云账号，您需要先注册阿里云账号。

开通阿里云百炼：使用阿里云主账号前往阿里云百炼大模型服务平台（北京或新加坡），阅读并同意协议后，将自动开通阿里云百炼，如果未弹出服务协议，则表示您已经开通。

如果开通服务时提示“您尚未进行实名认证”，请先进行实名认证。

403-Model.AccessDenied

Model access denied.

原因： 无权限调用对应的标准模型或自定义模型。

解决方案：

调用标准模型：使用子业务空间的API-KEY调用标准模型（例如qwen-plus）时，子业务空间需具备该模型的调用权限。详见模型调用授权。

调用自定义模型：自定义模型部署成功后，仅能用其所在业务空间的API-KEY调用，且无需模型调用授权。

403-App.AccessDenied

App access denied.

原因： 无权限访问应用或者模型。

解决方案：

仔细确认对访问的业务空间和子账号做了访问授权。

仔细检查应用是否发布。

仔细核实传入的APP ID、API KEY是否正确。

如果是Claude Code报错，请使用默认业务空间的API Key。

若上述建议都正确，建议刷新数据重新发布再调用，或尝试重新创建智能体。

403-Workspace.AccessDenied

Workspace access denied.

原因： 无权限访问业务空间的应用或者模型。

解决方案：

如果调用子业务空间的应用，请参考业务空间。

如果调用子业务空间的模型，请参考子业务空间的模型调用。

也可改为使用主账号的API KEY，主账号具有所有业务空间的权限。

403-AllocationQuota.FreeTierOnly

The free tier of the model has been exhausted. If you wish to continue access the model on a paid basis, please disable the "use free tier only" mode in the management console.

原因：开启了免费额度用完即停，且免费额度耗尽后发起请求。

控制台免费额度显示有小时级延迟。即使额度已用完，界面仍可能显示剩余额度。

解决方案：如需付费调用，请等待控制台显示免费额度用完后，关闭免费额度用完即停按钮。

404-ModelNotFound/model_not_found

The provided model xxx is not supported by the Batch API.

原因： 当前模型暂不支持 Batch 调用，或者可能存在模型名称拼写错误。

解决方案： 请参考OpenAI兼容-Batch，确认支持 Batch调用的模型及其正确名称。

Model can not be found. / The model xxx does not exist. / The model xxx does not exist or you do not have access to it.

原因： 当前访问的模型不存在，或您还未开通阿里云百炼服务。

解决方案：

请对照模型列表中的模型名称，检查您输入的模型名称（参数model的取值）是否正确。

请前往模型广场开通模型服务。

404-model_not_supported

Unsupported model xxx for OpenAI compatibility mode.

原因： 当前模型不支持以 OpenAI 兼容方式接入。

解决方案： 请您使用DashScope原生方式调用。

404-WorkSpaceNotFound

WorkSpace can not be found.

原因： 工作空间不存在。

404-NotFound

Not found!

原因：

要查询/操作的资源不存在。

使用定制热词时，传入的热词ID无效或对应热词不存在。

解决方案：

请检查要查询/操作的资源ID是否错误。

检查热词ID是否正确并参照API文档按照正确的方式进行调用。

409-Conflict

Model instance xxx already exists, please specify a suffix.

原因： 已存在重名的部署实例。

解决方案：为部署的模型指定不同的后缀名。

429-Throttling

Requests throttling triggered.

原因： 接口调用触发限流。

解决方案： 请降低调用频率或稍后重试。

Too many fine-tune job in running, please retry later. / Only 20 fine-tune job in running or succeeded allowed per user.

原因： 资源的创建触发平台限制。

解决方案： 可以删除不再使用的模型。如需提高并发量或保留更多模型，请发送邮件至modelstudio@service.aliyun.com申请提额。

Too many requests in route. Please try again later.

原因：请求过多触发限流。

解决方案：请稍后重试。

429-Throttling.RateQuota/LimitRequests/limit_requests

You have exceeded your request limit./Requests rate limit exceeded, please try again later. /You exceeded your current requests list.

原因： 调用频率（RPS/RPM）触发限流。

解决方案： 请参考限流，控制调用频率。

429-Throttling.BurstRate/limit_burst_rate

Request rate increased too quickly. To ensure system stability, please adjust your client logic to scale requests more smoothly over time.

原因：在未达到限流条件时，调用频率骤增，触发系统稳定性保护机制。

解决方案：建议优化客户端调用逻辑，采用平滑请求策略（如匀速调度、指数退避或请求队列缓冲），将请求均匀分散在时间窗口内，避免瞬时高峰。

429-Throttling.AllocationQuota/insufficient_quota

Allocated quota exceeded, please increase your quota limit./ You exceeded your current quota, please check your plan and billing details.

原因： 每秒钟或每分钟消耗Token数（TPS/TPM）触发限流。

解决方案： 前往限流文档查看模型限流条件并调整调用策略。

可参考限流FAQ避免触发限流。



Too many requests. Batch requests are being throttled due to system capacity limits. Please try again later.

原因： Batch请求过多触发限流。

解决方案： 暂时无法处理您的请求，请稍后再进行重试。

Free allocated quota exceeded.

原因： 免费额度已到期或耗尽，且该模型暂不支持按量计费。

解决方案： 使用其它模型替换，例如：通义千问Audio模型额度耗尽，可使用全模态模型。

429-CommodityNotPurchased

Commodity has not purchased yet.

原因： 业务空间未订购。

解决方案： 请先订购业务空间服务。

429-PrepaidBillOverdue

The prepaid bill is overdue.

原因： 业务空间预付费账单到期。

429-PostpaidBillOverdue

The postpaid bill is overdue.

原因： 模型推理商品已失效。

430-Audio.DecoderError

Decoder audio file failed.

原因：CosyVoice声音复刻音频文件解码失败。

解决方案：建议通过工具（如ffprobe、mediainfo）或命令（如Linux/macOS的file命令）确认音频文件的实际编码格式，以确保符合要求。

430-Audio.FileSizeExceed

File too large

原因： CosyVoice声音复刻音频文件大小超限。

解决方案： 用于声音复刻的音频文件需10M以内。

430-Audio.AudioRateError

File sample rate unsupported

原因： CosyVoice声音复刻音频文件采样率不支持。

解决方案： 采样率设置为16KHz及以上。

430-Audio.AudioSilentError

Silent file unsupported.

原因：CosyVoice声音复刻音频文件为静音或非静音长度过短。

解决方案： 音频时长应尽量控制在 10~15 秒之间，并包含至少一段超过 5 秒的连续语音。

500-InternalError/internal_error

An internal error has occured, please try again later or contact service support.

原因： 内部错误。

解决方案：

如果您使用（Qwen-Omni）模型，需要使用流式输出方式。

如果您使用CosyVoice声音复刻，则可能的原因是：

音频文件不规范，比如声音本身有问题，有杂音或者声音忽高忽低。请参见录音操作指南录音后重试。

录音文件URL无法访问，请按照CosyVoice声音复刻API中的说明操作后重试。

录音文件时长过长。尽量选择10~15秒的录音。录音时请确保朗读连贯，并包含至少一段超过 5 秒的连续语音。

Internal server error!

原因： 内部算法错误。

解决方案： 请稍后重试。

audio preprocess server error

使用CosyVoice声音复刻：

原因：音频文件不规范，比如声音本身有问题，有杂音或者声音忽高忽低。

解决方案：请参见录音操作指南录音后重试。

原因：录音文件URL无法访问，

解决方案：请按照CosyVoice声音复刻API中的说明操作后重试。

原因：录音文件时长过长。

解决方案：尽量选择10~15秒的录音。录音时请确保朗读连贯，并包含至少一段超过 5 秒的连续语音。

500-InternalError.FileUpload

oss upload error.

原因： 文件上传失败。

解决方案： 请检查OSS配置和网络。

500-InternalError.Upload

Failed to upload result.

原因： 生成结果上传失败。

解决方案： 请检查存储配置或稍后重试。

500-InternalError.Algo

inference internal error.

原因： 服务异常。

解决方案： 请先尝试重试，排除偶发情况。

Expecting ',' delimiter: line x column xxx (char xxx)

原因： 模型生成的JSON数据不合法，无法正常发起工具调用。

解决方案： 建议更换最新的模型或优化提示词后重试。

Missing Content-Length of multimodal url.

原因： URL请求的响应头信息缺失Content-Length字段。

解决方案： 如果问题无法解决，请尝试使用其他图片链接。

查看Content-Length字段

An error occurred in model serving, error message is: [Request rejected by inference engine!]

原因： 模型服务底层服务器出现错误。

解决方案： 请稍后重试。

An internal error has occured during algorithm execution.

原因： 算法运行时发生错误。

解决方案： 请稍后重试。

Inference error: Inference error.

原因： 推理发生错误。

解决方案： 请检查输入的图片文件是否有损坏或检查人物图片的质量（需包含完整清晰的人脸）。

Role must be in [user, assistant]

原因： 在使用Qwen-MT模型时，messages数组中包含了非 user角色的消息。

解决方案： 请确保messages数组中仅包含一个元素，且该元素必须是用户消息（User Message）。

Embedding_pipeline_Error: xxx

原因： 图像或视频预处理出错。

解决方案： 请确认上传的图片或视频及请求代码符合要求后重试。

Receive batching backend response failed!

原因： 服务内部错误。

解决方案： 请稍后重试。

An internal error has occured during execution, please try again later or contact service support. / algorithm process error. / inference error. / An internal error occurs during computation, please try this model later.

原因： 内部算法错误。

解决方案： 请稍后重试。

list index out of range

原因： messages 数组最后一位需为 User Message。

解决方案： 请调整messages数组的顺序，确保最后一个元素是 {"role": "user", ...}。

500-InternalError.Timeout

An internal timeout error has occured during execution, please try again later or contact service support.

原因： 异步任务提交后，在3小时内未返回结果，导致超时。

解决方案： 请检查任务执行情况，或联系技术支持。

500-SystemError

An system error has occured, please try again later.

原因： 系统错误。

解决方案： 请稍后重试。若您使用调用百炼应用，请参照示例代码或说明文档，查看是否代码编写有误，若依然无法确定问题，加入Spring AI Alibaba官网最下方提供的DING群，联系开发人员进行定位。

500-ModelServiceFailed

Failed to request model service.

原因： 模型服务调用失败。

解决方案： 请稍后重试。

500-RequestTimeOut

Request timed out, please try again later. / Response timeout! / I/O error on POST request for "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions": timeout

原因：

调用大模型时请求超时，超时报错时间为300秒。

使用语音识别（Paraformer）时，长时间未向服务器发送音频或者长时间发送静音音频。

解决方案：

通过流式输出方式发起请求，具体操作请参见流式输出。

将请求参数heartbeat设为true或及时结束识别任务。

调用通义千问模型时，响应体中会将已生成的内容返回，不再报超时错误。详情请参见文本生成。



500-InvokePluginFailed

Failed to invoke plugin.

原因： 插件调用失败。

解决方案： 请检查插件配置和可用性。

500-AppProcessFailed

Failed to proceed application request.

原因： 应用流程处理失败。

解决方案： 请检查应用配置和流程节点。

500-RewriteFailed

Failed to rewrite content for prompt.

原因： 调用改写prompt的大模型失败。

解决方案： 请稍后重试。

500-RetrivalFailed

Failed to retrieve data from documents.

原因： 文档检索失败。

解决方案： 请检查文档索引和检索配置。

503-ModelServingError

Too many requests. Your requests are being throttled due to system capacity limits. Please try again later.

原因： 网络资源目前处于饱和状态，暂时无法处理您的请求。

解决方案： 请稍后再进行尝试。

503-ModelUnavailable

Model is unavailable, please try again later.

原因： 模型暂时无法提供服务。

解决方案： 请稍后重试。

SDK 报错

error.AuthenticationError: No api key provided. You can set by dashscope.api_key = your_api_key in code, or you can set it via environment variable DASHSCOPE_API_KEY= your_api_key.

原因： 使用DashScope SDK 时未提供API Key。

解决方案： 具体配置API Key的方法，请参见配置API Key到环境变量。

openai.OpenAIError: The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable

原因： 使用 OpenAI SDK 时未传入 API Key。

解决方案：

通过环境变量传入 API Key 来源（推荐）

将DASHSCOPE_API_KEY设为环境变量（参见配置API Key到环境变量），初始化client时，通过os.getenv读取：

client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),...)

明文传入 API Key（仅限测试）

直接将 API Key 传入api_key参数：

client = OpenAI(api_key="sk-...", ...)

注意：此方法存在安全风险，请勿用于生产环境。

Bad Request for url: xxx

原因： 使用 Python requests 库时，添加 response.raise_for_status()语句导致报错时不返回服务端的具体错误内容。

解决方案： 请用 print(response.json()) 查看服务端返回信息。

Cannot resolve symbol 'ttsv2'

原因： 若您使用实时语音合成（CosyVoice），出现该问题的原因是DashScope SDK版本过低。

解决方案： 请安装最新版 DashScope SDK。

NetworkError

NoApiKeyException: Can not find api-key.

原因： 环境变量配置没有生效。

解决方案： 您可以重启客户端或IDE后重试。更多情况请参考常见问题。

ConnectException: Failed to connect to dashscope.aliyuncs.com

原因： 本地网络环境存在异常。

解决方案： 请检查本地网络，例如因证书问题导致无法访问 HTTPS，防火墙设置有误等情况。建议您更换网络环境或服务器进行测试。

InputRequiredException: Parameter invalid: text is null

原因：使用实时语音合成（CosyVoice）时未发送待合成文本。

解决方案：调用语音合成接口时为 text 参数赋值。

MultiModalConversation.call() missing 1 required positional argument: 'messages'

原因：当前使用的DashScope SDK版本过低。

解决方案：请安装最新版 DashScope SDK。

mismatched_model

The model 'xxx' for this request does not match the rest of the batch. Each batch must contain requests for a single model.

原因： 在单个 Batch 任务中，所有请求都必须选用同一个模型。

解决方案： 请根据输入文件检查您的输入文件。

duplicate_custom_id

The custom_id 'xxx' for this request is a duplicate of another request. The custom_id parameter must be unique for each request in a batch.

原因： 在单个 Batch 任务中，每条请求的 ID 必须唯一。

解决方案： 请根据输入文件检查您的输入文件，确保所有请求 ID 不重复。

Upload file capacity exceed limit. / Upload file number exceed limit.

原因： 上传文件失败，当前阿里云账号下的阿里云百炼存储空间已满或接近满额。

解决方案： 可以通过OpenAI兼容-File接口删除不需要的文件以释放空间。当前存储空间支持最大文件数为10000个，总量不超过100 GB。

WebSocket 报错

Invalid payload data

原因： 使用语音识别/翻译（Gummy）的WebSocket API，发送给服务端的JSON格式有误。

解决方案：

检查发送run-task指令时，payload中是否有“"input": {}”，若无，请添加。

确认在最终是否发送了完整的finish-task指令，且遵循其格式说明。请勿发送自创内容（如{ "input": { "end_of_stream": true } }）。

The decoded text message was too big for the output buffer and the endpoint does not support partial messages

原因： 使用语音识别（Paraformer）或语音识别/翻译（Gummy）的流式语音识别时，服务返回的识别结果数据量过大。

解决方案： 请分段发送待识别音频，建议每次发送的音频时长约为100毫秒，数据大小保持在1KB至16KB之间。

TimeoutError: websocket connection could not established within 5s. Please check your network connection, firewall settings, or server status.

原因： 若您使用语音合成（CosyVoice），无法在5秒内建立websocket连接。

解决方案： 请检查本地网络、防火墙设置，或更换网络环境或服务器进行测试。

unsupported audio format:xxx

原因： CosyVoice声音复刻时，上传的音频格式不符合模型要求。

解决方案： 音频格式需为 WAV（16bit）、MP3 或 M4A。请注意，不能仅凭文件后缀名判断格式，建议通过工具（如ffprobe、mediainfo）或命令（如Linux/macOS的file命令）确认音频文件的实际编码格式。

internal unknown error

原因： CosyVoice声音复刻音频文件格式可能不符合要求。

解决方案： 音频格式需为 WAV（16bit）、MP3 或 M4A。建议通过工具（如ffprobe、mediainfo）或命令确认音频文件的实际编码格式。

Invalid backend response received (missing status name)

原因： 使用语音识别（Paraformer）的录音文件识别的RESTful API时，请求参数拼写有误。

解决方案： 请参照API文档检查代码。

NO_INPUT_AUDIO_ERROR

原因： 未检测到有效语音。

解决方案： 若您使用语音识别（Paraformer）实时语音识别，请通过如下方式排查：

检查是否有音频输入。

检查音频格式是否正确（支持pcm、wav、mp3、opus、speex、aac、amr等）。

SUCCESS_WITH_NO_VALID_FRAGMENT

原因： 若您使用语音识别（Paraformer）录音文件识别，识别结果查询接口调用成功，但是VAD模块未检测到有效语音。

解决方案： 请排查录音文件是否包含有效语音，如果都是无效语音（例如纯静音），则没有识别结果是正常现象。

ASR_RESPONSE_HAVE_NO_WORDS

原因： 若您使用语音识别（Paraformer）录音文件识别，识别结果查询接口调用成功，但是最终识别结果为空。

解决方案： 请排查录音文件是否包含有效语音，或有效语音是否都是语气词且开启了顺滑参数disfluency_removal_enabled，导致语气词被过滤。

FILE_DOWNLOAD_FAILED

原因： 若您使用语音识别（Paraformer）录音文件识别，待识别文件下载失败。

解决方案： 请检查录音文件路径是否正确，以及是否可以通过外网访问和下载。

FILE_CHECK_FAILED

原因： 若您使用语音识别（Paraformer）录音文件识别，文件格式错误。

解决方案： 请检查录音文件是否是单轨/双轨的WAV格式或MP3格式。

FILE_TOO_LARGE

原因： 若您使用语音识别（Paraformer）录音文件识别，待识别文件过大。

解决方案： 请检查录音文件大小是否超过2GB，超过则需您对录音文件分段。

FILE_NORMALIZE_FAILED

原因： 若您使用语音识别（Paraformer）录音文件识别，待识别文件归一化失败。

解决方案： 请检查录音文件是否有损坏，是否可以正常播放。

FILE_PARSE_FAILED

原因： 若您使用语音识别（Paraformer）录音文件识别，文件解析失败。

解决方案： 请检查录音文件是否有损坏，是否可以正常播放。

MKV_PARSE_FAILED

原因： 若您使用语音识别（Paraformer）录音文件识别，MKV解析失败。

解决方案： 请检查录音文件是否损坏，是否可以正常播放。

FILE_TRANS_TASK_EXPIRED

原因： 若您使用语音识别（Paraformer）录音文件识别，录音文件识别任务过期。

解决方案： TaskId不存在，或者已过期。请重新提交任务。

REQUEST_INVALID_FILE_URL_VALUE

原因： 若您使用语音识别（Paraformer）录音文件识别，请求file_link参数非法。

解决方案： 请确认file_url参数格式是否正确。

CONTENT_LENGTH_CHECK_FAILED

原因： 若您使用语音识别（Paraformer）录音文件识别，content-length检查失败。

解决方案： 请检查下载待识别录音文件时，HTTP response中的content-length与文件实际大小是否一致。

FILE_404_NOT_FOUND

原因： 若您使用语音识别（Paraformer）录音文件识别，需要下载的文件不存在。

解决方案： 请检查文件URL是否正确。

FILE_403_FORBIDDEN

原因： 若您使用语音识别（Paraformer）录音文件识别，没有权限下载待识别录音。

解决方案： 请检查文件访问权限。

FILE_SERVER_ERROR

原因： 若您使用语音识别（Paraformer）录音文件识别，请求的文件所在的服务不可用。

解决方案： 请稍后重试或检查文件服务器状态。

AUDIO_DURATION_TOO_LONG

原因： 若您使用语音识别（Paraformer）录音文件识别，请求的文件时长超过12小时。

解决方案： 建议将音频进行切分，分多次提交识别任务。可使用FFmpeg等工具切分。

DECODE_ERROR

原因： 若您使用语音识别（Paraformer）录音文件识别，检测音频文件信息失败。

解决方案： 请确认文件下载链接中文件为支持的音频格式。

CLIENT_ERROR-[qwen-tts:]Engine return error code: 411

原因： 在进行Qwen-TTS实时语音合成时，选用的模型是qwen-tts-vc-realtime-2025-08-20，但音色是默认音色。该模型仅支持复刻音色。

解决方案： 请使用通过声音复刻生成的音色，而非默认音色。

NO_VALID_AUDIO_ERROR

原因： 使用语音识别（Paraformer）或语音识别/翻译（Gummy）时，待识别音频无效。 解决方案： 请检查音频格式、采样率等是否满足要求

指定工具调用方式

并行工具调用

单一城市的天气查询经过一次工具调用即可。如果输入问题需要调用多次工具，如“北京上海的天气如何”或“杭州天气，以及现在几点了”，发起 Function Calling 后只会返回一个工具调用信息，以提问“北京上海的天气如何”为例：



{

    "content": "",

    "refusal": null,

    "role": "assistant",

    "audio": null,

    "function_call": null,

    "tool_calls": [

        {

            "id": "call_61a2bbd82a8042289f1ff2",

            "function": {

                "arguments": "{\"location\": \"北京市\"}",

                "name": "get_current_weather"

            },

            "type": "function",

            "index": 0

        }

    ]

}

返回结果中只有北京市的入参信息。为了解决这一问题，在发起 Function Calling时，可设置请求参数parallel_tool_calls为true，这样返回对象中将包含所有需要调用的工具函数与入参信息。



### **tools** `array` （可选）

包含一个或多个工具对象的数组，供模型在 Function Calling 中调用。相关文档：Function Calling
使用 `tools` 时，必须将 `result_format` 设为 `message`。
发起 Function Calling，或提交工具执行结果时，都必须设置 `tools` 参数。

** 属性**

> 通过 HTTP 调用时，请将 **tools** 放入 **parameters** 对象中。暂时不支持 qwen-vl 与 qwen-audio 系列模型。

---

### **tool_choice** `string` 或 `object` （可选） 默认值为 `auto`

工具选择策略。若需对某类问题强制指定工具调用方式（例如始终使用某工具或禁用所有工具），可设置此参数。

* `auto`
大模型自主选择工具策略；
* `none`
若在特定请求中希望临时禁用工具调用，可设定 **tool_choice** 参数为 `none`；
* `{"type": "function", "function": {"name": "the_function_to_call"}}`
若希望强制调用某个工具，可设定 **tool_choice** 参数为 `{"type": "function", "function": {"name": "the_function_to_call"}}`，其中 `the_function_to_call` 是指定的工具函数名称。
> 思考模式的模型不支持强制调用某个工具。



Java SDK 中为 `toolChoice`。通过 HTTP 调用时，请将 **tool_choice** 放入 **parameters** 对象中。
