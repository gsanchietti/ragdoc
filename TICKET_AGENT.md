# RAGDoc Ticket Agent

The RAGDoc Ticket Agent is a specialized AI tool that processes individual Freshdesk support tickets into tutorial markdown files. It automatically fetches tickets, translates content if needed, analyzes the information, and generates structured tutorials.

## Key Features

- **Single Ticket Processing**: Focused approach for quality control and precision
- **Automatic Language Detection & Translation**: Handles non-English tickets seamlessly
- **AI-Powered Content Analysis**: Uses OpenAI to extract tutorial-worthy information
- **Quality Filtering**: Only processes tickets with sufficient content quality
- **Customizable Templates**: Flexible markdown output formatting
- **Environment Configuration**: Reads settings from `.env` file
- **Integration Ready**: Generated tutorials work with ragdoc fetch/indexing

## Quick Start

### 1. Configure Environment

Create or update your `.env` file:

```bash
# Required: Freshdesk API access
FRESHDESK_DOMAIN=yourcompany.freshdesk.com
FRESHDESK_API_KEY=your_api_key_here

# Required: OpenAI API for translation and analysis
OPENAI_API_KEY=sk-your_openai_key

# Optional: Customize models and paths
TICKET_AGENT_TRANSLATION_MODEL=gpt-4o-mini
TICKET_AGENT_ANALYSIS_MODEL=gpt-4o-mini
TICKET_AGENT_TEMPLATE=configs/tutorial_template.md
TICKET_AGENT_OUTPUT_DIR=generated_tutorials
```

### 2. Process a Ticket

Process a single ticket by ID:

```bash
ragdoc-ticket-agent 12345
```

### 3. Integrate with RAGDoc

After generating tutorials, add them to your RAGDoc knowledge base:

```bash
# Add tutorial directory to your configs/sources.yaml
# Then run:
ragdoc-fetch --config configs/sources.yaml
```

## Configuration Reference

### Environment Variables

All configuration can be set via environment variables in your `.env` file:

```bash
# Required
FRESHDESK_DOMAIN=yourcompany.freshdesk.com
FRESHDESK_API_KEY=your_api_key_here
OPENAI_API_KEY=sk-your_openai_key

# Optional (with defaults)
TICKET_AGENT_TRANSLATION_MODEL=gpt-4o-mini
TICKET_AGENT_ANALYSIS_MODEL=gpt-4o-mini
TICKET_AGENT_TEMPLATE=configs/tutorial_template.md
TICKET_AGENT_OUTPUT_DIR=generated_tutorials
```

### Command Line Options

```bash
ragdoc-ticket-agent TICKET_ID [OPTIONS]

Options:
  --debug                     Enable debug logging
  --freshdesk-domain DOMAIN   Override FRESHDESK_DOMAIN env var
  --freshdesk-api-key KEY     Override FRESHDESK_API_KEY env var
  --template PATH             Template file path
  --output-dir DIR            Output directory for tutorials
  --translation-model MODEL  AI model for translation
  --analysis-model MODEL     AI model for content analysis
```

## Examples

### Basic Usage
```bash
# Process a specific ticket
ragdoc-ticket-agent 12345

# Use custom template and output location
ragdoc-ticket-agent 12345 --template custom_template.md --output-dir tutorials/
```

### Advanced Usage
```bash
# Use different AI models with debug logging
ragdoc-ticket-agent 12345 \
  --translation-model gpt-4 \
  --analysis-model gpt-4 \
  --debug

# Override environment settings
ragdoc-ticket-agent 12345 \
  --freshdesk-domain mycompany \
  --template custom_tutorial.md
```

## Troubleshooting

### Common Issues

**"Failed to generate tutorial"**
- Check that the ticket exists and is accessible with your API key
- Verify the ticket is solved/closed (status 4 or 5)
- Ensure the ticket has sufficient content quality (score ≥ 6)

**"FRESHDESK_DOMAIN must be provided"**
- Add `FRESHDESK_DOMAIN=yourcompany.freshdesk.com` to your `.env` file
- Or use `--freshdesk-domain` command line option

**"OPENAI_API_KEY environment variable is required"**
- Add `OPENAI_API_KEY=sk-your_key` to your `.env` file
- Ensure your OpenAI API key is valid and has sufficient credits

### Debug Mode

Use `--debug` for detailed logging:

```bash
ragdoc-ticket-agent 12345 --debug
```

This will show:
- API request/response details
- Translation and analysis steps
- Quality scoring information
- File generation process
