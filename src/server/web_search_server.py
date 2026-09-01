# Web search MCP server
import os

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from typing import Annotated

# Import Zhipu AI client
from zai import ZhipuAiClient
load_dotenv()
# UTF-8 encoding format
def setup_encoding():
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

setup_encoding()
mcp = FastMCP("web_search")

@mcp.tool()
def web_search(
    search_query: Annotated[str, Field(description="搜索查询字符串。")],
    count: Annotated[int, Field(description="返回搜索结果数量，范围 1-50，默认 10。")] = 10,
    search_domain_filter: Annotated[str, Field(description="只搜索指定域名的内容，例如 'www.sohu.com'。")] = None,
    search_recency_filter: Annotated[str, Field(description="搜索时间范围过滤条件，默认 'noLimit'。")] = "noLimit",
    content_size: Annotated[str, Field(description="控制网页摘要长度，默认 'medium'。")] = "medium"
):
    """
    使用智谱 AI web_search API 执行网页搜索。

    参数名必须保持英文，例如 search_query、count、search_domain_filter。
    Returns:
        dict: 搜索结果，包含 title、url、summary、status、total 等字段。
    """
    try:
        # Get API key from environment variable or configuration
        api_key = os.getenv("ZHIPU_API_KEY")
        
        # Initialize Zhipu AI client
        client = ZhipuAiClient(api_key=api_key)
        
        # Perform web search
        response = client.web_search.web_search(
            search_engine="search_pro",
            search_query=search_query,
            count=count,
            search_domain_filter=search_domain_filter,
            search_recency_filter=search_recency_filter,
            content_size=content_size
        )
        
        # Process and return results
        if response:
            try:
                # Check if search_result attribute exists
                if hasattr(response, 'search_result'):
                    search_result = response.search_result
                    if search_result:
                        results = []
                        for item in search_result:
                            result_item = {
                                "title": item.title if hasattr(item, 'title') else "",
                                "url": item.link if hasattr(item, 'link') else (item.url if hasattr(item, 'url') else ""),
                                "summary": item.content if hasattr(item, 'content') else (item.summary if hasattr(item, 'summary') else "")
                            }
                            results.append(result_item)
                        return {
                            "status": "success",
                            "results": results,
                            "total": len(results)
                        }
                else:
                    return {
                        "status": "failed",
                        "message": "Response has no search_result attribute"
                    }
            except Exception as e:
                return {
                    "status": "failed",
                    "message": f"Error processing response: {str(e)}"
                }
        else:
            return {
                "status": "failed",
                "message": "No results returned from search API"
            }
            
    except Exception as e:
        return {
            "status": "failed",
            "message": f"Error during web search: {str(e)}"
        }

if __name__ == "__main__":
    mcp.run(transport='stdio')
