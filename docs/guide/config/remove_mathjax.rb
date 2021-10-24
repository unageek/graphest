# Copyright 2016-2021 The Khronos Group Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Derived from:
#   https://github.com/KhronosGroup/Vulkan-Docs/blob/v1.2.196/config/katex_replace.rb
#   https://github.com/KhronosGroup/Vulkan-Docs/blob/v1.2.196/config/katex_replace/extension.rb

require "asciidoctor/extensions"

Asciidoctor::Extensions.register do
  postprocessor RemoveMathJax
end

class RemoveMathJax < Asciidoctor::Extensions::Postprocessor
  MathJaXScript = /<script type="text\/x-mathjax-config">((?!<\/script>).)+<\/script>/m
  MathJaXCDN = /<script src="https:\/\/cdnjs.cloudflare.com\/ajax\/libs\/mathjax\/[0-9].[0-9].[0-9]\/MathJax.js\?config=[-_A-Za-z]+"><\/script>/m

  def process(document, output)
    if document.attr? "stem"
      output.sub! MathJaXScript, ""
      output.sub! MathJaXCDN, ""
    end
    output
  end
end
