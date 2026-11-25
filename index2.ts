import { OpenRouter } from '@openrouter/sdk';
import {
    type InputsUnion1,
    type OpenAIResponsesRefusalContent,
    type OpenResponsesResult,
    type OutputFunctionCallItem,
    type OutputMessageItem,
    type OutputReasoningItem,
    type ResponseOutputText,
    type ResponsesRequestToolFunction,
} from '@openrouter/sdk/models';

// 1. Give it access to a weather tool (2 step request) (15 minutes)
// 2. Give it access to the bash tool (2 step request)
// 3. Write the agent loop

/// 1. streaming (done), 2. web_search, 3. create_sub_agent (depth - 4)
/// context eng

const MAX_OUTPUT_CHARS = 12_000;
const MAX_STEPS = 20;

const truncate = (text: string) =>
    text.length > MAX_OUTPUT_CHARS
        ? `${text.slice(0, MAX_OUTPUT_CHARS)}\n...[truncated ${text.length - MAX_OUTPUT_CHARS} chars]`
        : text;

const bashExecute = async (command: string) => {
    console.log(`\n[bash] ${command}`);
    try {
        const result = await Bun.$`bash -lc ${command}`.cwd(__dirname).quiet();
        return {
            stdout: truncate(result.stdout.toString()),
            stderr: truncate(result.stderr.toString()),
        };
    } catch (e: any) {
        return {
            stdout: truncate(e?.stdout?.toString?.() ?? ''),
            stderr: truncate(e?.stderr?.toString?.() || e?.message || 'Error while running command'),
        };
    }
};

const client = new OpenRouter({
    apiKey: process.env.API_KEY,
});

function convertOutputItemToInput(
    item: OpenResponsesResult['output'][number],
): InputsUnion1 {
    if (isOutputMessageItem(item)) {
        const content = item.content.filter(isKnownMessageContent);
        if (content.length !== item.content.length) {
            throw new Error('Received an unsupported message content type');
        }
        return { ...item, content };
    }

    if (isFunctionCallOrReasoningItem(item)) {
        return item;
    }

    throw new Error(`Received an unexpected output item: ${item.type}`);
}

function isOutputMessageItem(
    item: OpenResponsesResult['output'][number],
): item is OutputMessageItem {
    return item.type === 'message';
}

function isKnownMessageContent(
    content: OutputMessageItem['content'][number],
): content is OpenAIResponsesRefusalContent | ResponseOutputText {
    return content.type === 'output_text' || content.type === 'refusal';
}

function isFunctionCallOrReasoningItem(
    item: OpenResponsesResult['output'][number],
): item is OutputFunctionCallItem | OutputReasoningItem {
    return item.type === 'function_call' || item.type === 'reasoning';
}

const inputs: InputsUnion1[] = [
    {
        content: 'Explore this repo to tell me what it does',
        role: 'user',
    },
];

const tools: ResponsesRequestToolFunction[] = [
    {
        type: 'function',
        name: 'bash',
        description: 'Runs a bash command',
        parameters: {
            type: 'object',
            properties: {
                command: {
                    type: 'string',
                    description: 'The bash command to run',
                },
            },
            required: ['command'],
        },
    },
];

for (let step = 0; step < MAX_STEPS; step++) {
    console.log('\n= = = = = = = = = = = = = = = = = = = = = = = =');

    const stream = await client.responses.send({
        responsesRequest: {
            model: 'openai/gpt-5.2',
            input: inputs,
            stream: true,
            tools,
        },
    });

    if (!(stream instanceof ReadableStream)) {
        throw new Error('Expected a streaming response');
    }

    let response: OpenResponsesResult | undefined;

    for await (const event of stream) {
        if (event.type === 'response.output_text.delta') {
            process.stdout.write(event.delta);
        } else if (event.type === 'response.completed') {
            response = event.response;
        } else if (event.type === 'response.failed') {
            throw new Error(event.response?.error?.message ?? 'Streaming response failed');
        }
    }

    if (!response) {
        throw new Error('Stream ended without a response.completed event');
    }

    const toolcalls = response.output.filter((x) => x.type === 'function_call');

    if (toolcalls.length === 0) {
        console.log();
        break;
    }

    inputs.push(...response.output.map(convertOutputItemToInput));

    for (const toolcall of toolcalls) {
        if (toolcall.name !== 'bash') {
            throw new Error(`Tool not found: ${toolcall.name}`);
        }

        const command = JSON.parse(toolcall.arguments).command;
        const bashResult = await bashExecute(command);
        inputs.push({
            type: 'function_call_output',
            callId: toolcall.callId,
            output: JSON.stringify(bashResult),
        });
    }
}