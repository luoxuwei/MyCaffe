// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: snapshot.proto

#include "snapshot.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>

PROTOBUF_PRAGMA_INIT_SEG

namespace _pb = ::PROTOBUF_NAMESPACE_ID;
namespace _pbi = _pb::internal;

namespace MyCaffe {
PROTOBUF_CONSTEXPR Snapshot_ParamBlok_ParamValue::Snapshot_ParamBlok_ParamValue(
    ::_pbi::ConstantInitialized)
  : value_(0){}
struct Snapshot_ParamBlok_ParamValueDefaultTypeInternal {
  PROTOBUF_CONSTEXPR Snapshot_ParamBlok_ParamValueDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~Snapshot_ParamBlok_ParamValueDefaultTypeInternal() {}
  union {
    Snapshot_ParamBlok_ParamValue _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 Snapshot_ParamBlok_ParamValueDefaultTypeInternal _Snapshot_ParamBlok_ParamValue_default_instance_;
PROTOBUF_CONSTEXPR Snapshot_ParamBlok::Snapshot_ParamBlok(
    ::_pbi::ConstantInitialized)
  : param_value_()
  , param_type_(&::_pbi::fixed_address_empty_string, ::_pbi::ConstantInitialized{})
  , layer_name_(&::_pbi::fixed_address_empty_string, ::_pbi::ConstantInitialized{})
  , kernel_n_(0)
  , kernel_c_(0)
  , kernel_h_(0)
  , kernel_w_(0){}
struct Snapshot_ParamBlokDefaultTypeInternal {
  PROTOBUF_CONSTEXPR Snapshot_ParamBlokDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~Snapshot_ParamBlokDefaultTypeInternal() {}
  union {
    Snapshot_ParamBlok _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 Snapshot_ParamBlokDefaultTypeInternal _Snapshot_ParamBlok_default_instance_;
PROTOBUF_CONSTEXPR Snapshot::Snapshot(
    ::_pbi::ConstantInitialized)
  : param_blok_(){}
struct SnapshotDefaultTypeInternal {
  PROTOBUF_CONSTEXPR SnapshotDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~SnapshotDefaultTypeInternal() {}
  union {
    Snapshot _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 SnapshotDefaultTypeInternal _Snapshot_default_instance_;
}  // namespace MyCaffe
static ::_pb::Metadata file_level_metadata_snapshot_2eproto[3];
static constexpr ::_pb::EnumDescriptor const** file_level_enum_descriptors_snapshot_2eproto = nullptr;
static constexpr ::_pb::ServiceDescriptor const** file_level_service_descriptors_snapshot_2eproto = nullptr;

const uint32_t TableStruct_snapshot_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::MyCaffe::Snapshot_ParamBlok_ParamValue, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::MyCaffe::Snapshot_ParamBlok_ParamValue, value_),
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::MyCaffe::Snapshot_ParamBlok, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::MyCaffe::Snapshot_ParamBlok, param_type_),
  PROTOBUF_FIELD_OFFSET(::MyCaffe::Snapshot_ParamBlok, layer_name_),
  PROTOBUF_FIELD_OFFSET(::MyCaffe::Snapshot_ParamBlok, kernel_n_),
  PROTOBUF_FIELD_OFFSET(::MyCaffe::Snapshot_ParamBlok, kernel_c_),
  PROTOBUF_FIELD_OFFSET(::MyCaffe::Snapshot_ParamBlok, kernel_h_),
  PROTOBUF_FIELD_OFFSET(::MyCaffe::Snapshot_ParamBlok, kernel_w_),
  PROTOBUF_FIELD_OFFSET(::MyCaffe::Snapshot_ParamBlok, param_value_),
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::MyCaffe::Snapshot, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::MyCaffe::Snapshot, param_blok_),
};
static const ::_pbi::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, -1, sizeof(::MyCaffe::Snapshot_ParamBlok_ParamValue)},
  { 7, -1, -1, sizeof(::MyCaffe::Snapshot_ParamBlok)},
  { 20, -1, -1, sizeof(::MyCaffe::Snapshot)},
};

static const ::_pb::Message* const file_default_instances[] = {
  &::MyCaffe::_Snapshot_ParamBlok_ParamValue_default_instance_._instance,
  &::MyCaffe::_Snapshot_ParamBlok_default_instance_._instance,
  &::MyCaffe::_Snapshot_default_instance_._instance,
};

const char descriptor_table_protodef_snapshot_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n\016snapshot.proto\022\007MyCaffe\"\223\002\n\010Snapshot\022/"
  "\n\nparam_blok\030\001 \003(\0132\033.MyCaffe.Snapshot.Pa"
  "ramBlok\032\325\001\n\tParamBlok\022\022\n\nparam_type\030\001 \001("
  "\t\022\022\n\nlayer_name\030\002 \001(\t\022\020\n\010kernel_n\030\003 \001(\005\022"
  "\020\n\010kernel_c\030\004 \001(\005\022\020\n\010kernel_h\030\005 \001(\005\022\020\n\010k"
  "ernel_w\030\006 \001(\005\022;\n\013param_value\030\007 \003(\0132&.MyC"
  "affe.Snapshot.ParamBlok.ParamValue\032\033\n\nPa"
  "ramValue\022\r\n\005value\030\001 \001(\001b\006proto3"
  ;
static ::_pbi::once_flag descriptor_table_snapshot_2eproto_once;
const ::_pbi::DescriptorTable descriptor_table_snapshot_2eproto = {
    false, false, 311, descriptor_table_protodef_snapshot_2eproto,
    "snapshot.proto",
    &descriptor_table_snapshot_2eproto_once, nullptr, 0, 3,
    schemas, file_default_instances, TableStruct_snapshot_2eproto::offsets,
    file_level_metadata_snapshot_2eproto, file_level_enum_descriptors_snapshot_2eproto,
    file_level_service_descriptors_snapshot_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::_pbi::DescriptorTable* descriptor_table_snapshot_2eproto_getter() {
  return &descriptor_table_snapshot_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY2 static ::_pbi::AddDescriptorsRunner dynamic_init_dummy_snapshot_2eproto(&descriptor_table_snapshot_2eproto);
namespace MyCaffe {

// ===================================================================

class Snapshot_ParamBlok_ParamValue::_Internal {
 public:
};

Snapshot_ParamBlok_ParamValue::Snapshot_ParamBlok_ParamValue(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor();
  // @@protoc_insertion_point(arena_constructor:MyCaffe.Snapshot.ParamBlok.ParamValue)
}
Snapshot_ParamBlok_ParamValue::Snapshot_ParamBlok_ParamValue(const Snapshot_ParamBlok_ParamValue& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  value_ = from.value_;
  // @@protoc_insertion_point(copy_constructor:MyCaffe.Snapshot.ParamBlok.ParamValue)
}

inline void Snapshot_ParamBlok_ParamValue::SharedCtor() {
value_ = 0;
}

Snapshot_ParamBlok_ParamValue::~Snapshot_ParamBlok_ParamValue() {
  // @@protoc_insertion_point(destructor:MyCaffe.Snapshot.ParamBlok.ParamValue)
  if (auto *arena = _internal_metadata_.DeleteReturnArena<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>()) {
  (void)arena;
    return;
  }
  SharedDtor();
}

inline void Snapshot_ParamBlok_ParamValue::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
}

void Snapshot_ParamBlok_ParamValue::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void Snapshot_ParamBlok_ParamValue::Clear() {
// @@protoc_insertion_point(message_clear_start:MyCaffe.Snapshot.ParamBlok.ParamValue)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  value_ = 0;
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* Snapshot_ParamBlok_ParamValue::_InternalParse(const char* ptr, ::_pbi::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::_pbi::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // double value = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 9)) {
          value_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<double>(ptr);
          ptr += sizeof(double);
        } else
          goto handle_unusual;
        continue;
      default:
        goto handle_unusual;
    }  // switch
  handle_unusual:
    if ((tag == 0) || ((tag & 7) == 4)) {
      CHK_(ptr);
      ctx->SetLastTag(tag);
      goto message_done;
    }
    ptr = UnknownFieldParse(
        tag,
        _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
        ptr, ctx);
    CHK_(ptr != nullptr);
  }  // while
message_done:
  return ptr;
failure:
  ptr = nullptr;
  goto message_done;
#undef CHK_
}

uint8_t* Snapshot_ParamBlok_ParamValue::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:MyCaffe.Snapshot.ParamBlok.ParamValue)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // double value = 1;
  static_assert(sizeof(uint64_t) == sizeof(double), "Code assumes uint64_t and double are the same size.");
  double tmp_value = this->_internal_value();
  uint64_t raw_value;
  memcpy(&raw_value, &tmp_value, sizeof(tmp_value));
  if (raw_value != 0) {
    target = stream->EnsureSpace(target);
    target = ::_pbi::WireFormatLite::WriteDoubleToArray(1, this->_internal_value(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::_pbi::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:MyCaffe.Snapshot.ParamBlok.ParamValue)
  return target;
}

size_t Snapshot_ParamBlok_ParamValue::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:MyCaffe.Snapshot.ParamBlok.ParamValue)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // double value = 1;
  static_assert(sizeof(uint64_t) == sizeof(double), "Code assumes uint64_t and double are the same size.");
  double tmp_value = this->_internal_value();
  uint64_t raw_value;
  memcpy(&raw_value, &tmp_value, sizeof(tmp_value));
  if (raw_value != 0) {
    total_size += 1 + 8;
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData Snapshot_ParamBlok_ParamValue::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    Snapshot_ParamBlok_ParamValue::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*Snapshot_ParamBlok_ParamValue::GetClassData() const { return &_class_data_; }

void Snapshot_ParamBlok_ParamValue::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to,
                      const ::PROTOBUF_NAMESPACE_ID::Message& from) {
  static_cast<Snapshot_ParamBlok_ParamValue *>(to)->MergeFrom(
      static_cast<const Snapshot_ParamBlok_ParamValue &>(from));
}


void Snapshot_ParamBlok_ParamValue::MergeFrom(const Snapshot_ParamBlok_ParamValue& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:MyCaffe.Snapshot.ParamBlok.ParamValue)
  GOOGLE_DCHECK_NE(&from, this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  static_assert(sizeof(uint64_t) == sizeof(double), "Code assumes uint64_t and double are the same size.");
  double tmp_value = from._internal_value();
  uint64_t raw_value;
  memcpy(&raw_value, &tmp_value, sizeof(tmp_value));
  if (raw_value != 0) {
    _internal_set_value(from._internal_value());
  }
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void Snapshot_ParamBlok_ParamValue::CopyFrom(const Snapshot_ParamBlok_ParamValue& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:MyCaffe.Snapshot.ParamBlok.ParamValue)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Snapshot_ParamBlok_ParamValue::IsInitialized() const {
  return true;
}

void Snapshot_ParamBlok_ParamValue::InternalSwap(Snapshot_ParamBlok_ParamValue* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(value_, other->value_);
}

::PROTOBUF_NAMESPACE_ID::Metadata Snapshot_ParamBlok_ParamValue::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_snapshot_2eproto_getter, &descriptor_table_snapshot_2eproto_once,
      file_level_metadata_snapshot_2eproto[0]);
}

// ===================================================================

class Snapshot_ParamBlok::_Internal {
 public:
};

Snapshot_ParamBlok::Snapshot_ParamBlok(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned),
  param_value_(arena) {
  SharedCtor();
  // @@protoc_insertion_point(arena_constructor:MyCaffe.Snapshot.ParamBlok)
}
Snapshot_ParamBlok::Snapshot_ParamBlok(const Snapshot_ParamBlok& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      param_value_(from.param_value_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  param_type_.InitDefault();
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    param_type_.Set("", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (!from._internal_param_type().empty()) {
    param_type_.Set(from._internal_param_type(), 
      GetArenaForAllocation());
  }
  layer_name_.InitDefault();
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    layer_name_.Set("", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (!from._internal_layer_name().empty()) {
    layer_name_.Set(from._internal_layer_name(), 
      GetArenaForAllocation());
  }
  ::memcpy(&kernel_n_, &from.kernel_n_,
    static_cast<size_t>(reinterpret_cast<char*>(&kernel_w_) -
    reinterpret_cast<char*>(&kernel_n_)) + sizeof(kernel_w_));
  // @@protoc_insertion_point(copy_constructor:MyCaffe.Snapshot.ParamBlok)
}

inline void Snapshot_ParamBlok::SharedCtor() {
param_type_.InitDefault();
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  param_type_.Set("", GetArenaForAllocation());
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
layer_name_.InitDefault();
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  layer_name_.Set("", GetArenaForAllocation());
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
::memset(reinterpret_cast<char*>(this) + static_cast<size_t>(
    reinterpret_cast<char*>(&kernel_n_) - reinterpret_cast<char*>(this)),
    0, static_cast<size_t>(reinterpret_cast<char*>(&kernel_w_) -
    reinterpret_cast<char*>(&kernel_n_)) + sizeof(kernel_w_));
}

Snapshot_ParamBlok::~Snapshot_ParamBlok() {
  // @@protoc_insertion_point(destructor:MyCaffe.Snapshot.ParamBlok)
  if (auto *arena = _internal_metadata_.DeleteReturnArena<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>()) {
  (void)arena;
    return;
  }
  SharedDtor();
}

inline void Snapshot_ParamBlok::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  param_type_.Destroy();
  layer_name_.Destroy();
}

void Snapshot_ParamBlok::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void Snapshot_ParamBlok::Clear() {
// @@protoc_insertion_point(message_clear_start:MyCaffe.Snapshot.ParamBlok)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  param_value_.Clear();
  param_type_.ClearToEmpty();
  layer_name_.ClearToEmpty();
  ::memset(&kernel_n_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&kernel_w_) -
      reinterpret_cast<char*>(&kernel_n_)) + sizeof(kernel_w_));
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* Snapshot_ParamBlok::_InternalParse(const char* ptr, ::_pbi::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::_pbi::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // string param_type = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 10)) {
          auto str = _internal_mutable_param_type();
          ptr = ::_pbi::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(ptr);
          CHK_(::_pbi::VerifyUTF8(str, "MyCaffe.Snapshot.ParamBlok.param_type"));
        } else
          goto handle_unusual;
        continue;
      // string layer_name = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 18)) {
          auto str = _internal_mutable_layer_name();
          ptr = ::_pbi::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(ptr);
          CHK_(::_pbi::VerifyUTF8(str, "MyCaffe.Snapshot.ParamBlok.layer_name"));
        } else
          goto handle_unusual;
        continue;
      // int32 kernel_n = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 24)) {
          kernel_n_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // int32 kernel_c = 4;
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 32)) {
          kernel_c_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // int32 kernel_h = 5;
      case 5:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 40)) {
          kernel_h_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // int32 kernel_w = 6;
      case 6:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 48)) {
          kernel_w_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // repeated .MyCaffe.Snapshot.ParamBlok.ParamValue param_value = 7;
      case 7:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 58)) {
          ptr -= 1;
          do {
            ptr += 1;
            ptr = ctx->ParseMessage(_internal_add_param_value(), ptr);
            CHK_(ptr);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<58>(ptr));
        } else
          goto handle_unusual;
        continue;
      default:
        goto handle_unusual;
    }  // switch
  handle_unusual:
    if ((tag == 0) || ((tag & 7) == 4)) {
      CHK_(ptr);
      ctx->SetLastTag(tag);
      goto message_done;
    }
    ptr = UnknownFieldParse(
        tag,
        _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
        ptr, ctx);
    CHK_(ptr != nullptr);
  }  // while
message_done:
  return ptr;
failure:
  ptr = nullptr;
  goto message_done;
#undef CHK_
}

uint8_t* Snapshot_ParamBlok::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:MyCaffe.Snapshot.ParamBlok)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // string param_type = 1;
  if (!this->_internal_param_type().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_param_type().data(), static_cast<int>(this->_internal_param_type().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "MyCaffe.Snapshot.ParamBlok.param_type");
    target = stream->WriteStringMaybeAliased(
        1, this->_internal_param_type(), target);
  }

  // string layer_name = 2;
  if (!this->_internal_layer_name().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_layer_name().data(), static_cast<int>(this->_internal_layer_name().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "MyCaffe.Snapshot.ParamBlok.layer_name");
    target = stream->WriteStringMaybeAliased(
        2, this->_internal_layer_name(), target);
  }

  // int32 kernel_n = 3;
  if (this->_internal_kernel_n() != 0) {
    target = stream->EnsureSpace(target);
    target = ::_pbi::WireFormatLite::WriteInt32ToArray(3, this->_internal_kernel_n(), target);
  }

  // int32 kernel_c = 4;
  if (this->_internal_kernel_c() != 0) {
    target = stream->EnsureSpace(target);
    target = ::_pbi::WireFormatLite::WriteInt32ToArray(4, this->_internal_kernel_c(), target);
  }

  // int32 kernel_h = 5;
  if (this->_internal_kernel_h() != 0) {
    target = stream->EnsureSpace(target);
    target = ::_pbi::WireFormatLite::WriteInt32ToArray(5, this->_internal_kernel_h(), target);
  }

  // int32 kernel_w = 6;
  if (this->_internal_kernel_w() != 0) {
    target = stream->EnsureSpace(target);
    target = ::_pbi::WireFormatLite::WriteInt32ToArray(6, this->_internal_kernel_w(), target);
  }

  // repeated .MyCaffe.Snapshot.ParamBlok.ParamValue param_value = 7;
  for (unsigned i = 0,
      n = static_cast<unsigned>(this->_internal_param_value_size()); i < n; i++) {
    const auto& repfield = this->_internal_param_value(i);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
        InternalWriteMessage(7, repfield, repfield.GetCachedSize(), target, stream);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::_pbi::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:MyCaffe.Snapshot.ParamBlok)
  return target;
}

size_t Snapshot_ParamBlok::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:MyCaffe.Snapshot.ParamBlok)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated .MyCaffe.Snapshot.ParamBlok.ParamValue param_value = 7;
  total_size += 1UL * this->_internal_param_value_size();
  for (const auto& msg : this->param_value_) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(msg);
  }

  // string param_type = 1;
  if (!this->_internal_param_type().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_param_type());
  }

  // string layer_name = 2;
  if (!this->_internal_layer_name().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_layer_name());
  }

  // int32 kernel_n = 3;
  if (this->_internal_kernel_n() != 0) {
    total_size += ::_pbi::WireFormatLite::Int32SizePlusOne(this->_internal_kernel_n());
  }

  // int32 kernel_c = 4;
  if (this->_internal_kernel_c() != 0) {
    total_size += ::_pbi::WireFormatLite::Int32SizePlusOne(this->_internal_kernel_c());
  }

  // int32 kernel_h = 5;
  if (this->_internal_kernel_h() != 0) {
    total_size += ::_pbi::WireFormatLite::Int32SizePlusOne(this->_internal_kernel_h());
  }

  // int32 kernel_w = 6;
  if (this->_internal_kernel_w() != 0) {
    total_size += ::_pbi::WireFormatLite::Int32SizePlusOne(this->_internal_kernel_w());
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData Snapshot_ParamBlok::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    Snapshot_ParamBlok::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*Snapshot_ParamBlok::GetClassData() const { return &_class_data_; }

void Snapshot_ParamBlok::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to,
                      const ::PROTOBUF_NAMESPACE_ID::Message& from) {
  static_cast<Snapshot_ParamBlok *>(to)->MergeFrom(
      static_cast<const Snapshot_ParamBlok &>(from));
}


void Snapshot_ParamBlok::MergeFrom(const Snapshot_ParamBlok& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:MyCaffe.Snapshot.ParamBlok)
  GOOGLE_DCHECK_NE(&from, this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  param_value_.MergeFrom(from.param_value_);
  if (!from._internal_param_type().empty()) {
    _internal_set_param_type(from._internal_param_type());
  }
  if (!from._internal_layer_name().empty()) {
    _internal_set_layer_name(from._internal_layer_name());
  }
  if (from._internal_kernel_n() != 0) {
    _internal_set_kernel_n(from._internal_kernel_n());
  }
  if (from._internal_kernel_c() != 0) {
    _internal_set_kernel_c(from._internal_kernel_c());
  }
  if (from._internal_kernel_h() != 0) {
    _internal_set_kernel_h(from._internal_kernel_h());
  }
  if (from._internal_kernel_w() != 0) {
    _internal_set_kernel_w(from._internal_kernel_w());
  }
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void Snapshot_ParamBlok::CopyFrom(const Snapshot_ParamBlok& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:MyCaffe.Snapshot.ParamBlok)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Snapshot_ParamBlok::IsInitialized() const {
  return true;
}

void Snapshot_ParamBlok::InternalSwap(Snapshot_ParamBlok* other) {
  using std::swap;
  auto* lhs_arena = GetArenaForAllocation();
  auto* rhs_arena = other->GetArenaForAllocation();
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  param_value_.InternalSwap(&other->param_value_);
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &param_type_, lhs_arena,
      &other->param_type_, rhs_arena
  );
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &layer_name_, lhs_arena,
      &other->layer_name_, rhs_arena
  );
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(Snapshot_ParamBlok, kernel_w_)
      + sizeof(Snapshot_ParamBlok::kernel_w_)
      - PROTOBUF_FIELD_OFFSET(Snapshot_ParamBlok, kernel_n_)>(
          reinterpret_cast<char*>(&kernel_n_),
          reinterpret_cast<char*>(&other->kernel_n_));
}

::PROTOBUF_NAMESPACE_ID::Metadata Snapshot_ParamBlok::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_snapshot_2eproto_getter, &descriptor_table_snapshot_2eproto_once,
      file_level_metadata_snapshot_2eproto[1]);
}

// ===================================================================

class Snapshot::_Internal {
 public:
};

Snapshot::Snapshot(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned),
  param_blok_(arena) {
  SharedCtor();
  // @@protoc_insertion_point(arena_constructor:MyCaffe.Snapshot)
}
Snapshot::Snapshot(const Snapshot& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      param_blok_(from.param_blok_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  // @@protoc_insertion_point(copy_constructor:MyCaffe.Snapshot)
}

inline void Snapshot::SharedCtor() {
}

Snapshot::~Snapshot() {
  // @@protoc_insertion_point(destructor:MyCaffe.Snapshot)
  if (auto *arena = _internal_metadata_.DeleteReturnArena<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>()) {
  (void)arena;
    return;
  }
  SharedDtor();
}

inline void Snapshot::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
}

void Snapshot::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void Snapshot::Clear() {
// @@protoc_insertion_point(message_clear_start:MyCaffe.Snapshot)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  param_blok_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* Snapshot::_InternalParse(const char* ptr, ::_pbi::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::_pbi::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // repeated .MyCaffe.Snapshot.ParamBlok param_blok = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 10)) {
          ptr -= 1;
          do {
            ptr += 1;
            ptr = ctx->ParseMessage(_internal_add_param_blok(), ptr);
            CHK_(ptr);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<10>(ptr));
        } else
          goto handle_unusual;
        continue;
      default:
        goto handle_unusual;
    }  // switch
  handle_unusual:
    if ((tag == 0) || ((tag & 7) == 4)) {
      CHK_(ptr);
      ctx->SetLastTag(tag);
      goto message_done;
    }
    ptr = UnknownFieldParse(
        tag,
        _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
        ptr, ctx);
    CHK_(ptr != nullptr);
  }  // while
message_done:
  return ptr;
failure:
  ptr = nullptr;
  goto message_done;
#undef CHK_
}

uint8_t* Snapshot::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:MyCaffe.Snapshot)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .MyCaffe.Snapshot.ParamBlok param_blok = 1;
  for (unsigned i = 0,
      n = static_cast<unsigned>(this->_internal_param_blok_size()); i < n; i++) {
    const auto& repfield = this->_internal_param_blok(i);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
        InternalWriteMessage(1, repfield, repfield.GetCachedSize(), target, stream);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::_pbi::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:MyCaffe.Snapshot)
  return target;
}

size_t Snapshot::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:MyCaffe.Snapshot)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated .MyCaffe.Snapshot.ParamBlok param_blok = 1;
  total_size += 1UL * this->_internal_param_blok_size();
  for (const auto& msg : this->param_blok_) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(msg);
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData Snapshot::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    Snapshot::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*Snapshot::GetClassData() const { return &_class_data_; }

void Snapshot::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to,
                      const ::PROTOBUF_NAMESPACE_ID::Message& from) {
  static_cast<Snapshot *>(to)->MergeFrom(
      static_cast<const Snapshot &>(from));
}


void Snapshot::MergeFrom(const Snapshot& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:MyCaffe.Snapshot)
  GOOGLE_DCHECK_NE(&from, this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  param_blok_.MergeFrom(from.param_blok_);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void Snapshot::CopyFrom(const Snapshot& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:MyCaffe.Snapshot)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Snapshot::IsInitialized() const {
  return true;
}

void Snapshot::InternalSwap(Snapshot* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  param_blok_.InternalSwap(&other->param_blok_);
}

::PROTOBUF_NAMESPACE_ID::Metadata Snapshot::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_snapshot_2eproto_getter, &descriptor_table_snapshot_2eproto_once,
      file_level_metadata_snapshot_2eproto[2]);
}

// @@protoc_insertion_point(namespace_scope)
}  // namespace MyCaffe
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::MyCaffe::Snapshot_ParamBlok_ParamValue*
Arena::CreateMaybeMessage< ::MyCaffe::Snapshot_ParamBlok_ParamValue >(Arena* arena) {
  return Arena::CreateMessageInternal< ::MyCaffe::Snapshot_ParamBlok_ParamValue >(arena);
}
template<> PROTOBUF_NOINLINE ::MyCaffe::Snapshot_ParamBlok*
Arena::CreateMaybeMessage< ::MyCaffe::Snapshot_ParamBlok >(Arena* arena) {
  return Arena::CreateMessageInternal< ::MyCaffe::Snapshot_ParamBlok >(arena);
}
template<> PROTOBUF_NOINLINE ::MyCaffe::Snapshot*
Arena::CreateMaybeMessage< ::MyCaffe::Snapshot >(Arena* arena) {
  return Arena::CreateMessageInternal< ::MyCaffe::Snapshot >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
