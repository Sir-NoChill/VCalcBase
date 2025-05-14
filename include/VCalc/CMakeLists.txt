set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls -dialect=vcalc)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -dialect=vcalc)
mlir_tablegen(Types.h.inc -gen-typedef-decls -dialect=vcalc)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs -dialect=vcalc)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=vcalc)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=vcalc)

##### NOTE These only apply if you have defined attributes or enums ####
mlir_tablegen(Attrs.h.inc -gen-attrdef-decls -attrdefs-dialect=vcalc)
mlir_tablegen(Attrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=vcalc)
mlir_tablegen(Enums.h.inc -gen-enum-decls -dialect=vcalc)
mlir_tablegen(Enums.cpp.inc -gen-enum-defs -dialect=vcalc)
add_public_tablegen_target(VCalcOpsIncGen)

# Seperate pass generation seems to be the MLIR best practice
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -dialect=vcalc -name=VCalcToStandard)
add_public_tablegen_target(VCalcPassesIncGen)
